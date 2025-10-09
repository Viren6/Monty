use std::{collections::BTreeSet, io};

#[cfg(target_os = "linux")]
use std::fs;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadBinding {
    pub cpu: usize,
    pub node: Option<usize>,
}

impl ThreadBinding {
    #[inline]
    pub fn apply(&self) {
        #[cfg(target_os = "linux")]
        {
            if let Err(err) = set_current_thread_affinity(self.cpu) {
                // Binding can legitimately fail when running without the required
                // privileges or when the requested CPU is no longer part of the
                // affinity mask. We silently ignore the error in release builds
                // to keep the engine functional in such environments while still
                // exposing potential issues during debugging.
                debug_assert!(false, "failed to bind thread to cpu {}: {}", self.cpu, err);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ThreadTopology {
    allowed_bindings: Vec<ThreadBinding>,
    worker_bindings: Vec<ThreadBinding>,
}

impl ThreadTopology {
    pub fn detect(requested_threads: usize) -> Self {
        let requested_threads = requested_threads.max(1);
        let allowed_cpus = detect_allowed_cpus();

        let nodes = discover_numa_nodes(&allowed_cpus);
        let mut allowed_bindings = Vec::new();
        let mut seen = BTreeSet::new();

        for node in nodes {
            for cpu in node.cpus {
                if seen.insert(cpu) {
                    allowed_bindings.push(ThreadBinding {
                        cpu,
                        node: Some(node.id),
                    });
                }
            }
        }

        for &cpu in &allowed_cpus {
            if seen.insert(cpu) {
                allowed_bindings.push(ThreadBinding { cpu, node: None });
            }
        }

        if allowed_bindings.is_empty() {
            let fallback_count = requested_threads;
            allowed_bindings = (0..fallback_count)
                .map(|cpu| ThreadBinding { cpu, node: None })
                .collect();
        } else {
            allowed_bindings
                .sort_by_key(|binding| (binding.node.unwrap_or(usize::MAX), binding.cpu));
        }

        let mut worker_bindings = Vec::with_capacity(requested_threads);
        for idx in 0..requested_threads {
            let binding = allowed_bindings[idx % allowed_bindings.len()];
            worker_bindings.push(binding);
        }

        Self {
            allowed_bindings,
            worker_bindings,
        }
    }

    #[inline]
    pub fn worker_count(&self) -> usize {
        self.worker_bindings.len()
    }

    #[inline]
    pub fn binding_for_worker(&self, worker: usize) -> Option<ThreadBinding> {
        if self.worker_bindings.is_empty() {
            None
        } else {
            Some(self.worker_bindings[worker % self.worker_bindings.len()])
        }
    }

    #[inline]
    pub fn init_binding_count(&self) -> usize {
        self.allowed_bindings.len().max(1)
    }

    #[inline]
    pub fn init_binding(&self, idx: usize) -> Option<ThreadBinding> {
        if self.allowed_bindings.is_empty() {
            None
        } else {
            Some(self.allowed_bindings[idx % self.allowed_bindings.len()])
        }
    }
}

#[derive(Clone, Debug)]
struct NumaNode {
    id: usize,
    cpus: Vec<usize>,
}

fn discover_numa_nodes(allowed: &[usize]) -> Vec<NumaNode> {
    #[cfg(target_os = "linux")]
    {
        let mut nodes = Vec::new();
        let allowed_set: BTreeSet<_> = allowed.iter().copied().collect();

        if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if !name.starts_with("node") {
                    continue;
                }

                let id = match name[4..].parse::<usize>() {
                    Ok(id) => id,
                    Err(_) => continue,
                };

                let path = entry.path().join("cpulist");
                let Ok(contents) = fs::read_to_string(path) else {
                    continue;
                };

                let mut cpus = parse_cpu_list(&contents);
                cpus.retain(|cpu| allowed_set.contains(cpu));

                if cpus.is_empty() {
                    continue;
                }

                cpus.sort_unstable();
                nodes.push(NumaNode { id, cpus });
            }
        }

        nodes.sort_by_key(|node| node.id);
        return nodes;
    }

    #[allow(unreachable_code)]
    Vec::new()
}

fn detect_allowed_cpus() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            if let Some(line) = status
                .lines()
                .find(|line| line.starts_with("Cpus_allowed_list:"))
            {
                if let Some(list) = line.split(':').nth(1) {
                    let cpus = parse_cpu_list(list);
                    if !cpus.is_empty() {
                        return cpus;
                    }
                }
            }
        }
    }

    let fallback = std::thread::available_parallelism()
        .map(|nz| nz.get())
        .unwrap_or(1);
    (0..fallback).collect()
}

fn parse_cpu_list(list: &str) -> Vec<usize> {
    let mut cpus = Vec::new();

    for part in list.split(',') {
        let entry = part.trim();
        if entry.is_empty() {
            continue;
        }

        if let Some((start, end)) = entry.split_once('-') {
            if let (Ok(start), Ok(end)) =
                (start.trim().parse::<usize>(), end.trim().parse::<usize>())
            {
                if start <= end {
                    cpus.extend(start..=end);
                }
            }
            continue;
        }

        if let Ok(single) = entry.parse::<usize>() {
            cpus.push(single);
        }
    }

    cpus
}

#[cfg(target_os = "linux")]
fn set_current_thread_affinity(cpu: usize) -> io::Result<()> {
    use std::mem::size_of;

    let bits_per_word = size_of::<libc::c_ulong>() * 8;
    let words = cpu / bits_per_word + 1;
    let mut mask = vec![0 as libc::c_ulong; words];
    let idx = cpu / bits_per_word;
    let bit = cpu % bits_per_word;
    mask[idx] |= 1 << bit;

    let res = unsafe {
        libc::sched_setaffinity(
            0,
            (mask.len() * size_of::<libc::c_ulong>()) as libc::size_t,
            mask.as_ptr() as *const libc::cpu_set_t,
        )
    };

    if res != 0 {
        return Err(io::Error::last_os_error());
    }

    Ok(())
}