
fn main() {
    // Old params
    let knight_val = 437;
    let bishop_val = 409;
    let rook_val = 768;
    let queen_val = 1512;
    let material_offset = 559;
    let material_div1 = 36;
    let material_div2 = 1226;

    // Target: Minimize MSE of (Score_Old - Score_New)
    
    let mut best_mse = f32::MAX;
    let mut best_offset = 0;
    let mut best_scale = 0.0;
    
    // Search range
    for offset in (0..20000).step_by(100) {
        for scale_int in 0..2000 {
            let scale = scale_int as f32 / 1000000.0; // 0.0 to 0.002
            
            let mut mse = 0.0;
            let mut count = 0;
            
            // Iterate over material range
            for mat in (0..8000).step_by(100) {
                // Old Logic
                let mat_scaled_old = material_offset + mat / material_div1;
                let cp_factor_old = mat_scaled_old as f32 / material_div2 as f32;
                
                // Raw values (example)
                let raw_win = 0.4;
                let raw_draw = 0.4;
                let raw_loss = 0.2;
                
                let raw_wdl = EvalWdl::new(raw_win, raw_draw, raw_loss);
                let cp_base = raw_wdl.to_cp_i32();
                let cp_old = (cp_base as f32 * cp_factor_old) as i32;
                let score_old_dampened = 1.0 / (1.0 + (-(cp_old as f32) / 400.0).exp());
                
                // New Logic
                let draw_adj = raw_draw * (offset - mat) as f32 * scale;
                let sum = raw_win + raw_draw + draw_adj + raw_loss;
                let wdl_new = EvalWdl {
                    win: raw_win / sum,
                    draw: (raw_draw + draw_adj) / sum,
                    loss: raw_loss / sum,
                };
                let score_new = wdl_new.score();
                
                let diff = score_old_dampened - score_new;
                mse += diff * diff;
                count += 1;
            }
            
            if mse < best_mse {
                best_mse = mse;
                best_offset = offset;
                best_scale = scale;
            }
        }
    }
    
    println!("Best Params: Offset={}, Scale={:.6}, MSE={:.6}", best_offset, best_scale, best_mse);
    
    // Print verification for best params
    println!("Mat, Score_Old, Score_New, Diff");
    let offset = best_offset;
    let scale = best_scale;
    for mat in (0..8000).step_by(500) {
        let mat_scaled_old = material_offset + mat / material_div1;
        let cp_factor_old = mat_scaled_old as f32 / material_div2 as f32;
        
        let raw_win = 0.4;
        let raw_draw = 0.4;
        let raw_loss = 0.2;
        
        let raw_wdl = EvalWdl::new(raw_win, raw_draw, raw_loss);
        let cp_base = raw_wdl.to_cp_i32();
        let cp_old = (cp_base as f32 * cp_factor_old) as i32;
        let score_old = 1.0 / (1.0 + (-(cp_old as f32) / 400.0).exp());
        
        let draw_adj = raw_draw * (offset - mat) as f32 * scale;
        let sum = raw_win + raw_draw + draw_adj + raw_loss;
        let wdl_new = EvalWdl {
            win: raw_win / sum,
            draw: (raw_draw + draw_adj) / sum,
            loss: raw_loss / sum,
        };
        let score_new = wdl_new.score();
        
        println!("{}, {:.4}, {:.4}, {:.4}", mat, score_old, score_new, score_old - score_new);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EvalWdl {
    pub win: f32,
    pub draw: f32,
    pub loss: f32,
}

impl EvalWdl {
    pub fn new(win: f32, draw: f32, loss: f32) -> Self {
        let mut win = win.clamp(0.0, 1.0);
        let mut draw = draw.clamp(0.0, 1.0);
        let mut loss = loss.clamp(0.0, 1.0);

        let sum = win + draw + loss;

        if sum <= 0.0 {
            return Self {
                win: 1.0 / 3.0,
                draw: 1.0 / 3.0,
                loss: 1.0 / 3.0,
            };
        }

        let inv = 1.0 / sum;
        win *= inv;
        draw *= inv;
        loss *= inv;

        Self { win, draw, loss }
    }

    pub fn score(&self) -> f32 {
        self.win + 0.5 * self.draw
    }

    pub fn from_draw_and_score(draw: f32, score: f32) -> Self {
        let draw = draw.clamp(0.0, 1.0);
        let min_score = draw * 0.5;
        let max_score = 1.0 - draw * 0.5;
        let score = score.clamp(min_score.min(max_score), max_score.max(min_score));

        let win = (score - draw * 0.5).max(0.0);
        let loss = (1.0 - draw - win).max(0.0);
        Self::new(win, draw, loss)
    }

    pub fn to_cp_i32(&self) -> i32 {
        const K: f32 = 400.0;
        let score = self.score().clamp(0.0, 1.0);
        if score <= 0.0 || score >= 1.0 { return 0; }
        (-K * (1.0 / score - 1.0).ln()) as i32
    }
}
