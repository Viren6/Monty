#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadBinding {
    cpu: usize,
    node: usize,
}

impl ThreadBinding {
    #[must_use]
    pub fn cpu(&self) -> usize {
        self.cpu
    }

    #[must_use]
    pub fn node(&self) -> usize {
        self.node
    }
}

pub fn thread_bindings(count: usize) -> Vec<Option<ThreadBinding>> {
    platform::thread_bindings(count)
}

pub fn bind_to(binding: Option<ThreadBinding>) {
    platform::bind_to(binding);
}

#[cfg(target_os = "linux")]
mod platform {
    use super::ThreadBinding;
    use once_cell::sync::Lazy;
    use std::{
        fs, io,
        path::{Path, PathBuf},
    };

    static TOPOLOGY: Lazy<NumaTopology> = Lazy::new(NumaTopology::detect);

    pub fn thread_bindings(count: usize) -> Vec<Option<ThreadBinding>> {
        TOPOLOGY.bindings(count)
    }

    pub fn bind_to(binding: Option<ThreadBinding>) {
        if let Some(binding) = binding {
            let _ = set_thread_affinity(binding.cpu());
        }
    }

    fn set_thread_affinity(cpu: usize) -> io::Result<()> {
        unsafe {
            let mut mask: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut mask);
            libc::CPU_SET(cpu, &mut mask);

            let result = libc::pthread_setaffinity_np(
                libc::pthread_self(),
                std::mem::size_of::<libc::cpu_set_t>(),
                &mask,
            );

            if result != 0 {
                Err(io::Error::from_raw_os_error(result))
            } else {
                Ok(())
            }
        }
    }

    #[derive(Default)]
    struct NumaTopology {
        cpus: Vec<(usize, usize)>,
    }

    impl NumaTopology {
        fn detect() -> Self {
            let mut topology = NumaTopology::default();

            let path = Path::new("/sys/devices/system/node");
            let Ok(entries) = fs::read_dir(path) else {
                return topology;
            };

            let mut nodes = Vec::new();

            for entry in entries.flatten() {
                let name = entry.file_name();
                let Some(name) = name.to_str() else {
                    continue;
                };

                let Some(id_str) = name.strip_prefix("node") else {
                    continue;
                };

                let Ok(id) = id_str.parse::<usize>() else {
                    continue;
                };

                let cpulist_path = entry.path().join("cpulist");
                if let Some(cpus) = parse_cpu_list(&cpulist_path) {
                    if !cpus.is_empty() {
                        nodes.push((id, cpus));
                    }
                }
            }

            nodes.sort_by_key(|(id, _)| *id);

            for (node, cpus) in nodes {
                for cpu in cpus {
                    topology.cpus.push((node, cpu));
                }
            }

            topology
        }

        fn bindings(&self, count: usize) -> Vec<Option<ThreadBinding>> {
            if count == 0 {
                return Vec::new();
            }

            if self.cpus.is_empty() {
                return vec![None; count];
            }

            (0..count)
                .map(|idx| {
                    let source = if idx < self.cpus.len() {
                        idx
                    } else {
                        idx % self.cpus.len()
                    };
                    let (node, cpu) = self.cpus[source];
                    Some(ThreadBinding { node, cpu })
                })
                .collect()
        }
    }

    fn parse_cpu_list(path: &PathBuf) -> Option<Vec<usize>> {
        let data = fs::read_to_string(path).ok()?;
        let mut cpus = Vec::new();

        for part in data
            .split(|c| matches!(c, ',' | '\n' | '\r' | '\t'))
            .map(str::trim)
        {
            if part.is_empty() {
                continue;
            }

            if let Some((start, end)) = part.split_once('-') {
                let Ok(start) = start.trim().parse::<usize>() else {
                    continue;
                };
                let Ok(end) = end.trim().parse::<usize>() else {
                    continue;
                };

                if end < start {
                    continue;
                }

                cpus.extend(start..=end);
            } else if let Ok(cpu) = part.parse::<usize>() {
                cpus.push(cpu);
            }
        }

        Some(cpus)
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::ThreadBinding;

    pub fn thread_bindings(count: usize) -> Vec<Option<ThreadBinding>> {
        vec![None; count]
    }

    pub fn bind_to(_binding: Option<ThreadBinding>) {}
}