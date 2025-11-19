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

use crate::chess::EvalWdl;

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
    let guard = SYZYGY.read().ok()?;
    let tablebase = guard.as_ref()?;

    if position.occ().count_ones() > 7 {
        return None;
    }

    let fen = position.as_fen();
    let chess: Chess = Fen::from_ascii(fen.as_bytes())
        .ok()?
        .into_position(CastlingMode::Standard)
        .ok()?;

    let wdl = tablebase.probe_wdl_after_zeroing(&chess).ok()?;
    Some(eval_from_wdl(wdl))
}

fn eval_from_wdl(wdl: Wdl) -> EvalWdl {
    match wdl {
        Wdl::Win => EvalWdl::new(1.0, 0.0, 0.0),
        Wdl::Loss => EvalWdl::new(0.0, 0.0, 1.0),

        // everything that should be treated as a draw:
        Wdl::CursedWin | Wdl::Draw | Wdl::BlessedLoss => EvalWdl::new(0.0, 1.0, 0.0),
    }
}
