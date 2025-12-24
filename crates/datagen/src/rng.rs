use std::time::{SystemTime, UNIX_EPOCH};

pub struct Rand(pub u32);

impl Default for Rand {
    fn default() -> Self {
        Self(
            (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("valid")
                .as_nanos()
                & 0xFFFF_FFFF) as u32,
        )
    }
}

impl Rand {
    pub fn rand_int(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    pub fn with_seed() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Guaranteed increasing.")
            .as_micros() as u32;

        Self(seed)
    }
    pub fn rand_float(&mut self) -> f32 {
        (self.rand_int() as f32) / (u32::MAX as f32 + 1.0)
    }

    pub fn rand_std_normal(&mut self) -> f32 {
        let u1 = self.rand_float();
        let u2 = self.rand_float();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn sample_gamma(&mut self, alpha: f32) -> f32 {
        if alpha < 1.0 {
            let u = self.rand_float();
            return self.sample_gamma(1.0 + alpha) * u.powf(1.0 / alpha);
        }

        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let z = self.rand_std_normal();
            let v_term = 1.0 + c * z;
            if v_term <= 0.0 {
                continue;
            }
            let v = v_term * v_term * v_term;
            let u = self.rand_float();

            let x = d * v;
            if u < 1.0 - 0.0331 * z * z * z * z {
                return x;
            }
            if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
                return x;
            }
        }
    }
}
