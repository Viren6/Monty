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
    root_pst: f32 = 3.133, 1.0, 10.0, 0.364, 0.002;
    depth_2_pst: f32 = 1.189, 1.0, 10.0, 0.04, 0.002;
    winning_pst_threshold: f32 = 0.610, 0.0, 1.0, 0.07, 0.002;
    winning_pst_max: f32 = 1.618, 0.1, 10.0, 0.15, 0.002;
    root_cpuct: f32 = 0.405, 0.1, 5.0, 0.0314, 0.002;
    cpuct: f32 = 0.261, 0.1, 5.0, 0.0314, 0.002;
    cpuct_var_weight: f32 = 0.801, 0.0, 2.0, 0.0851, 0.002;
    cpuct_var_scale: f32 = 0.275, 0.0, 2.0, 0.0257, 0.002;
    cpuct_visits_scale: f32 = 36.779, 1.0, 512.0, 0.373, 0.002;
    expl_tau: f32 = 0.675, 0.1, 1.0, 0.0623, 0.002;
    gini_base: f32 = 0.461, 0.2, 2.0, 0.0679, 0.002;
    gini_ln_multiplier: f32 = 1.536, 0.4, 3.0, 0.1634, 0.002;
    gini_min: f32 = 2.256, 0.5, 4.0, 0.21, 0.002;
    knight_value: i32 = 438, 250, 750, 45, 0.002;
    bishop_value: i32 = 399, 250, 750, 45, 0.002;
    rook_value: i32 = 746, 400, 1000, 65, 0.002;
    queen_value: i32 = 1508, 900, 1600, 125, 0.002;
    material_offset: i32 = 575, 400, 1200, 70, 0.002;
    material_div1: i32 = 37, 16, 64, 3, 0.002;
    material_div2: i32 = 1215, 512, 1536, 102, 0.002;
    tm_opt_value1: f64 = 0.630, 0.1, 1.2, 0.0686, 0.002;
    tm_opt_value2: f64 = 0.428, 0.1, 1.0, 0.0392, 0.002;
    tm_opt_value3: f64 = 0.669, 0.1, 1.2, 0.0822, 0.002;
    tm_optscale_value1: f64 = 1.643, 0.1, 2.0, 0.1271, 0.002;
    tm_optscale_value2: f64 = 2.359, 0.1, 5.0, 0.2510, 0.002;
    tm_optscale_value3: f64 = 0.491, 0.1, 1.0, 0.0499, 0.002;
    tm_optscale_value4: f64 = 0.258, 0.1, 1.0, 0.0240, 0.002;
    tm_max_value1: f64 = 2.953, 1.0, 10.0, 0.3072, 0.002;
    tm_max_value2: f64 = 2.763, 1.0, 10.0, 0.2928, 0.002;
    tm_max_value3: f64 = 2.756, 1.0, 10.0, 0.2843, 0.002;
    tm_maxscale_value1: f64 = 13.705, 1.0, 24.0, 1.1357, 0.002;
    tm_maxscale_value2: f64 = 5.023, 1.0, 12.0, 0.3691, 0.002;
    tm_bonus_ply: f64 = 11.417, 1.0, 30.0, 1.072, 0.002;
    tm_bonus_value1: f64 = 0.473, 0.1, 2.0, 0.0488, 0.002;
    tm_max_time: f64 = 0.903, 0.400, 0.990, 0.0837, 0.002;
    tm_mtg: i32 = 27, 10, 60, 3, 0.002;
    tm_falling_eval1: f32 = 0.054, 0.0, 0.2, 0.0055, 0.002;
    tm_falling_eval2: f32 = 0.719, 0.1, 1.0, 0.0648, 0.002;
    tm_falling_eval3: f32 = 1.622, 0.1, 3.0, 0.1644, 0.002;
    tm_bmi1: f32 = 0.255, 0.1, 1.0, 0.0305, 0.002;
    tm_bmi2: f32 = 0.874, 0.1, 2.0, 0.0948, 0.002;
    tm_bmi3: f32 = 3.259, 0.1, 6.4, 0.3121, 0.002;
    tm_bmv1: f32 = 3.619, 0.1, 5.0, 0.2696, 0.002;
    tm_bmv2: f32 = 0.351, 0.1, 1.0, 0.0309, 0.002;
    tm_bmv3: f32 = 0.499, 0.1, 1.0, 0.0519, 0.002;
    tm_bmv4: f32 = 2.643, 0.1, 8.0, 0.3946, 0.002;
    tm_bmv5: f32 = 0.632, 0.1, 1.0, 0.0523, 0.002;
    tm_bmv6: f32 = 1.888, 0.1, 3.0, 0.1560, 0.002;
}
