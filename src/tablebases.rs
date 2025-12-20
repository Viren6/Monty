use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    sync::RwLock,
};

use montyformat::chess::Position as MontyPosition;
use once_cell::sync::Lazy;
use shakmaty::{fen::Fen, CastlingMode, Chess};
use shakmaty_syzygy::{Tablebase, Wdl};

use crate::chess::{ChessState, EvalWdl, GameState, Move};

static SYZYGY: Lazy<RwLock<Option<Tablebase<Chess>>>> = Lazy::new(|| RwLock::new(None));

#[derive(Clone, Copy, Debug, Default)]
pub struct SyzygySummary {
    pub directories: usize,
    pub files: usize,
}

impl SyzygySummary {
    pub const fn disabled() -> Self {
        Self {
            directories: 0,
            files: 0,
        }
    }

    pub const fn is_enabled(&self) -> bool {
        self.files > 0
    }
}

pub fn configure_syzygy(path: Option<&str>) -> Result<SyzygySummary, String> {
    let mut guard = SYZYGY
        .write()
        .map_err(|_| "failed to configure Syzygy tablebases".to_string())?;

    let Some(path) = path.map(str::trim).filter(|p| !p.is_empty()) else {
        *guard = None;
        return Ok(SyzygySummary::disabled());
    };

    let mut tablebase = Tablebase::<Chess>::new();
    let mut summary = SyzygySummary::default();
    let os_value = OsString::from(path);

    for dir in env::split_paths(&os_value) {
        if dir.as_os_str().is_empty() {
            continue;
        }

        let directories = directories_including_subdirectories(&dir)?;
        for directory in directories {
            let files = tablebase.add_directory(&directory).map_err(|err| {
                format!("failed to read Syzygy path {}: {err}", directory.display())
            })?;
            summary.directories += 1;
            summary.files += files;
        }
    }

    if summary.directories == 0 {
        return Err("SyzygyPath did not contain any directories".to_string());
    }

    *guard = Some(tablebase);
    Ok(summary)
}

fn directories_including_subdirectories(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut stack = vec![root.to_path_buf()];
    let mut directories = Vec::new();

    while let Some(dir) = stack.pop() {
        let metadata = fs::metadata(&dir)
            .map_err(|err| format!("failed to read Syzygy path {}: {err}", dir.display()))?;

        if !metadata.is_dir() {
            continue;
        }

        directories.push(dir.clone());

        for entry in fs::read_dir(&dir)
            .map_err(|err| format!("failed to read Syzygy path {}: {err}", dir.display()))?
        {
            let entry = entry
                .map_err(|err| format!("failed to read Syzygy path {}: {err}", dir.display()))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            }
        }
    }

    Ok(directories)
}

pub fn probe_wdl(position: &MontyPosition) -> Option<EvalWdl> {
    probe_wdl_state(position).map(|tb| tb.eval)
}

#[derive(Clone, Copy, Debug)]
pub struct TablebaseWdl {
    pub eval: EvalWdl,
    pub state: GameState,
}

pub fn probe_wdl_state(position: &MontyPosition) -> Option<TablebaseWdl> {
    let guard = SYZYGY.read().ok()?;
    let tablebase = guard.as_ref()?;

    if position.occ().count_ones() > 7 {
        return None;
    }

    let chess = monty_to_shakmaty(position)?;

    let wdl = tablebase.probe_wdl_after_zeroing(&chess).ok()?;
    Some(TablebaseWdl {
        eval: eval_from_wdl(wdl),
        state: wdl_to_state(wdl),
    })
}

fn eval_from_wdl(wdl: Wdl) -> EvalWdl {
    match wdl {
        Wdl::Win => EvalWdl::new(1.0, 0.0, 0.0),
        Wdl::Loss => EvalWdl::new(0.0, 0.0, 1.0),

        // everything that should be treated as a draw:
        Wdl::CursedWin | Wdl::Draw | Wdl::BlessedLoss => EvalWdl::new(0.0, 1.0, 0.0),
    }
}

fn wdl_to_state(wdl: Wdl) -> GameState {
    match wdl {
        Wdl::Win => GameState::Won(0),
        Wdl::Loss => GameState::Lost(0),
        Wdl::CursedWin | Wdl::Draw | Wdl::BlessedLoss => GameState::Draw,
    }
}

fn monty_to_shakmaty(position: &MontyPosition) -> Option<Chess> {
    let fen = position.as_fen();
    Fen::from_ascii(fen.as_bytes())
        .ok()?
        .into_position(CastlingMode::Standard)
        .ok()
}

pub fn probe_root_dtz_move(state: &ChessState) -> Option<(Move, EvalWdl)> {
    let guard = SYZYGY.read().ok()?;
    let tablebase = guard.as_ref()?;

    if state.board().occ().count_ones() > 7 {
        return None;
    }

    let root_chess = monty_to_shakmaty(&state.board())?;
    let root_dtz = tablebase.probe_dtz(&root_chess).ok()?;
    let root_wdl = tablebase.probe_wdl_after_zeroing(&root_chess).ok()?;

    let prefer_low = root_dtz.ignore_rounding().is_positive();
    let mut best: Option<(Move, i32)> = None;

    state.map_legal_moves(|mov| {
        let mut child = state.clone();
        child.make_move(mov);

        if let Some(chess) = monty_to_shakmaty(&child.board()) {
            if let Ok(dtz) = tablebase.probe_dtz(&chess) {
                let val = i32::from(dtz.ignore_rounding());

                match best {
                    None => best = Some((mov, val)),
                    Some((_, current)) => {
                        let better = if prefer_low { val < current } else { val > current };
                        if better {
                            best = Some((mov, val));
                        }
                    }
                }
            }
        }
    });

    let (mov, _) = best?;
    Some((mov, eval_from_wdl(root_wdl)))
}
