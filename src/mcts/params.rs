#[derive(Clone)]
struct Param<T> {
    val: T,
    min: T,
    max: T,
}

impl<T> Param<T> {
    fn new(val: T, min: T, max: T) -> Self {
        Self { val, min, max }
    }
}

impl Param<i32> {
    fn set(&mut self, val: i32) {
        self.val = val.clamp(self.min, self.max);
    }

    fn info(&self, name: &str) {
        println!(
            "option name {} type spin default {:.0} min {:.0} max {:.0}",
            name, self.val, self.min, self.max,
        );
    }

    fn list(&self, name: &str, step: i32, r: f32) {
        println!(
            "{}, {}, {}, {}, {}, {}",
            name, self.val, self.min, self.max, step, r,
        );
    }
}

impl Param<f32> {
    fn set(&mut self, val: i32) {
        let actual = val as f32 / 1000.0;
        self.val = actual.clamp(self.min, self.max);
    }

    fn info(&self, name: &str) {
        println!(
            "option name {} type spin default {:.0} min {:.0} max {:.0}",
            name,
            self.val * 1000.0,
            self.min * 1000.0,
            self.max * 1000.0,
        );
    }

    fn list(&self, name: &str, step: f32, r: f32) {
        println!(
            "{}, {}, {}, {}, {}, {}",
            name,
            self.val * 1000.0,
            self.min * 1000.0,
            self.max * 1000.0,
            step * 1000.0,
            r,
        );
    }
}

impl Param<f64> {
    fn set(&mut self, val: i32) {
        let actual = val as f64 / 1000.0;
        self.val = actual.clamp(self.min, self.max);
    }

    fn info(&self, name: &str) {
        println!(
            "option name {} type spin default {:.0} min {:.0} max {:.0}",
            name,
            self.val * 1000.0,
            self.min * 1000.0,
            self.max * 1000.0,
        );
    }

    fn list(&self, name: &str, step: f64, r: f64) {
        println!(
            "{}, {}, {}, {}, {}, {}",
            name,
            self.val * 1000.0,
            self.min * 1000.0,
            self.max * 1000.0,
            step * 1000.0,
            r,
        );
    }
}

macro_rules! make_mcts_params {
    ($($name:ident: $t:ty = $val:expr, $min:expr, $max:expr, $step:expr, $r:expr;)*) => {
        #[derive(Clone)]
        pub struct MctsParams {
            $($name: Param<$t>,)*
        }

        impl Default for MctsParams {
            fn default() -> Self {
                Self {
                    $($name: Param::new($val, $min, $max),)*
                }
            }
        }

        impl MctsParams {
        $(
            pub fn $name(&self) -> $t {
                self.$name.val
            }
        )*

            pub fn info(self) {
                $(self.$name.info(stringify!($name));)*
            }

            pub fn set(&mut self, name: &str, val: i32) {
                match name {
                    $(stringify!($name) => self.$name.set(val),)*
                    _ => println!("unknown option!"),
                }
            }

            pub fn list_spsa(&self) {
                $(self.$name.list(stringify!($name), $step, $r);)*
            }
        }
    };
}

make_mcts_params! {
    root_pst_adjustment: f32 = 0.34, 0.01, 1.0, 0.034, 0.002;
    depth_pst_adjustment: f32 = 1.8, 0.1, 10.0, 0.18, 0.002;
    winning_pst_threshold: f32 = 0.603, 0.0, 1.0, 0.05, 0.002;
    winning_pst_max: f32 = 1.615, 0.1, 10.0, 0.1, 0.002;
    base_pst_adjustment: f32 = 0.1, 0.01, 1.0, 0.01, 0.002;
    root_cpuct: f32 = if cfg!(feature = "datagen") { 1.0 } else { 0.422 }, 0.1, 5.0, 0.065, 0.002;
    cpuct:      f32 = if cfg!(feature = "datagen") { 0.157 } else { 0.269 }, 0.1, 5.0, 0.065, 0.002;
    cpuct_var_weight: f32 = 0.808, 0.0, 2.0, 0.085, 0.002;
    cpuct_var_scale: f32 = 0.278, 0.0, 2.0, 0.02, 0.002;
    cpuct_var_warmup: f32 = 0.5, 0.0, 1.0, 0.01, 0.002;
    cpuct_visits_scale: f32 = 36.91, 1.0, 512.0, 3.2, 0.002;
    expl_tau: f32 = 0.676, 0.1, 1.0, 0.05, 0.002;
    gini_base: f32 = 0.463, 0.2, 2.0, 0.0679, 0.002;
    gini_ln_multiplier: f32 = 1.567, 0.4, 3.0, 0.1634, 0.002;
    gini_min: f32 = 2.26, 0.5, 4.0, 0.21, 0.002;
    sharpness_scale: f32 = 2.449, 0.0, 5.0, 0.1, 0.002;
    sharpness_quadratic: f32 = 0.872, -5.0, 5.0, 0.1, 0.002;
    tm_hard_limit: f64 = 0.55, 0.1, 1.0, 0.05, 0.002;
    tm_opt_base: f64 = 0.04, 0.01, 1.0, 0.005, 0.002;
    tm_mtg: i32 = 25, 1, 100, 2, 0.002;
    tm_feval_scale: f32 = 0.05, 0.0, 1.0, 0.005, 0.002;
    tm_feval_max: f32 = 1.5, 1.0, 5.0, 0.1, 0.002;
    tm_bmi_scale: f32 = 0.1, 0.0, 1.0, 0.01, 0.002;
    tm_bmi_max: f32 = 2.0, 1.0, 5.0, 0.1, 0.002;
    tm_bmv_scale: f32 = 0.5, 0.0, 2.0, 0.05, 0.002;
    tm_bmv_max: f32 = 2.0, 1.0, 5.0, 0.1, 0.002;
    butterfly_reduction_factor: i32 = 8192, 1, 65536, 819, 0.002;
    butterfly_policy_divisor: i32 = 16384, 1, 131072, 1638, 0.002;
    policy_top_p: f32 = 0.7, 0.1, 1.0, 0.05, 0.002;
    min_policy_actions: i32 = 6, 1, 32, 1, 0.002;
    visit_threshold_power: i32 = 3, 0, 8, 1, 0.002;
    virtual_loss_weight: f64 = 2.5, 1.0, 5.0, 0.25, 0.002;
    contempt: i32 = 0, -1000, 1000, 10, 0.0; //Do not tune this value!
}
