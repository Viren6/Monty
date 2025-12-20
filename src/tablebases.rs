use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    sync::RwLock,
};

use montyformat::chess::Position as MontyPosition;
use once_cell::sync::Lazy;
use shakmaty::{fen::Fen, CastlingMode, Chess, Position as _, uci::UciMove};
use shakmaty_syzygy::{Dtz, Tablebase, Wdl};

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

fn tablebase_guard() -> Option<std::sync::RwLockReadGuard<'static, Option<Tablebase<Chess>>>> {
    SYZYGY.read().ok()
}

fn to_chess(position: &MontyPosition) -> Option<Chess> {
    let fen = position.as_fen();
    Fen::from_ascii(fen.as_bytes())
        .ok()?
        .into_position(CastlingMode::Standard)
        .ok()
}

fn probe_wdl_inner(position: &MontyPosition) -> Option<Wdl> {
    let guard = tablebase_guard()?;
    let tablebase = guard.as_ref()?;

    if position.occ().count_ones() > 7 {
        return None;
    }

    let chess = to_chess(position)?;
    tablebase.probe_wdl_after_zeroing(&chess).ok()
}

pub fn probe_wdl(position: &MontyPosition) -> Option<EvalWdl> {
    probe_wdl_inner(position).map(eval_from_wdl)
}

pub fn probe_wdl_with_state(position: &MontyPosition) -> Option<(EvalWdl, GameState)> {
    let wdl = probe_wdl_inner(position)?;
    let eval = eval_from_wdl(wdl);

    let state = match wdl {
        Wdl::Win => GameState::Won(0),
        Wdl::Loss => GameState::Lost(0),
        _ => GameState::Draw,
    };

    Some((eval, state))
}

pub fn probe_root_dtz_best_move(state: &ChessState) -> Option<(Move, Dtz)> {
    let guard = tablebase_guard()?;
    let tablebase = guard.as_ref()?;

    if state.board().occ().count_ones() > 7 {
        return None;
    }

    let chess = to_chess(&state.board())?;
    let root_dtz = tablebase.probe_dtz(&chess).ok()?.ignore_rounding();
    let target_sign = root_dtz.signum();

    let mut legal_moves = Vec::new();
    state.map_legal_moves(|mov| legal_moves.push(mov));

    let mut matching: Vec<(Move, Dtz)> = Vec::new();
    let mut fallback: Vec<(Move, Dtz)> = Vec::new();

    for mov in legal_moves {
        let uci = mov.to_uci(&state.castling());
        let Ok(uci_move) = UciMove::from_ascii(uci.as_bytes()) else {
            continue;
        };

        let Ok(smove) = uci_move.to_move::<Chess>(&chess) else {
            continue;
        };

        let mut after = chess.clone();
        after.play_unchecked(smove);

        let dtz = match tablebase.probe_dtz(&after) {
            Ok(v) => v.ignore_rounding(),
            Err(_) => continue,
        };

        let our_dtz = Dtz(-dtz.0);
        if our_dtz.signum() == target_sign {
            matching.push((mov, our_dtz));
        } else {
            fallback.push((mov, our_dtz));
        }
    }

    let select_from = if !matching.is_empty() {
        matching
    } else {
        fallback
    };

    if target_sign > 0 {
        select_from
            .into_iter()
            .min_by_key(|(_, dtz)| dtz.0)
    } else {
        select_from
            .into_iter()
            .max_by_key(|(_, dtz)| dtz.0)
    }
}

fn eval_from_wdl(wdl: Wdl) -> EvalWdl {
    match wdl {
        Wdl::Win => EvalWdl::new(1.0, 0.0, 0.0),
        Wdl::Loss => EvalWdl::new(0.0, 0.0, 1.0),

        // everything that should be treated as a draw:
        Wdl::CursedWin | Wdl::Draw | Wdl::BlessedLoss => EvalWdl::new(0.0, 1.0, 0.0),
    }
}
