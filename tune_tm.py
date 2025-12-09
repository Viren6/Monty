import math
import random

# Original Parameters
tm_opt_value1 = 0.64
tm_opt_value2 = 0.434
tm_opt_value3 = 0.66
tm_optscale_value1 = 1.645
tm_optscale_value2 = 2.476
tm_optscale_value3 = 0.483
tm_optscale_value4 = 0.26
tm_max_value1 = 2.877
tm_max_value2 = 2.85
tm_max_value3 = 2.717
tm_maxscale_value1 = 13.275
tm_maxscale_value2 = 5.141
tm_bonus_ply = 11.475
tm_bonus_value1 = 0.452
tm_max_time = 0.881
tm_mtg = 28

def get_time_original(time, increment, ply):
    inc = increment if increment is not None else 0
    mtg = float(tm_mtg)
    
    time_left = max(1.0, time + inc * (mtg - 1) - 10 * (2 + mtg))
    log_time = math.log10(time_left / 1000.0)
    
    opt_constant = min(tm_opt_value3 / 100.0, 
                       tm_opt_value1 / 100.0 + tm_opt_value2 / 1000.0 * log_time)
    
    opt_scale = min(tm_optscale_value4 * time / time_left,
                    tm_optscale_value1 / 100.0 + 
                    math.pow(ply + tm_optscale_value2, tm_optscale_value3) * opt_constant)
                    
    max_constant = max(tm_max_value3, 
                       tm_max_value1 + tm_max_value2 * log_time)
                       
    max_scale = min(tm_maxscale_value2, 
                    max_constant + ply / tm_maxscale_value1)
                    
    bonus = 1.0
    if ply < tm_bonus_ply:
        bonus = 1.0 + math.log10(tm_bonus_ply - ply) * tm_bonus_value1
        
    opt_time = opt_scale * bonus * time_left
    max_time = min(time * tm_max_time, max_scale * opt_time)
    
    return opt_time, max_time

def get_time_simple(params, time, increment, ply):
    (
        c_base, c_ply_mult, c_ply_pow, 
        bonus_val, bonus_ply,
        max_base, max_ply_mult
    ) = params
    
    inc = increment if increment is not None else 0
    mtg = float(tm_mtg)
    
    time_left = max(1.0, time + inc * (mtg - 1) - 10 * (2 + mtg))
    
    opt_frac = c_base + c_ply_mult * math.pow(ply, c_ply_pow)
    
    bonus = 1.0
    if ply < bonus_ply:
        bonus = 1.0 + (bonus_ply - ply) * bonus_val
        
    opt_time = opt_frac * bonus * time_left
    
    max_factor = max_base + ply * max_ply_mult
    max_time = min(time * 0.8, opt_time * max_factor)
    
    return opt_time, max_time

def objective(params):
    error = 0.0
    count = 0
    
    times = [1000, 10000, 60000, 300000]
    incs = [0, 100, 1000, 5000]
    plies = range(0, 150, 5)
    
    for t in times:
        for i in incs:
            for p in plies:
                orig_opt, orig_max = get_time_original(t, i, p)
                simp_opt, simp_max = get_time_simple(params, t, i, p)
                
                # Log Error with penalty for under-usage
                diff_opt = math.log(simp_opt + 1) - math.log(orig_opt + 1)
                diff_max = math.log(simp_max + 1) - math.log(orig_max + 1)
                
                # Penalize under-usage more (factor of 50)
                if diff_opt < 0:
                    error += (diff_opt ** 2) * 50.0
                else:
                    error += diff_opt ** 2
                    
                if diff_max < 0:
                    error += (diff_max ** 2) * 50.0
                else:
                    error += diff_max ** 2
                    
                count += 1
                
    return error / count

# Manual Params Test
# c_base, c_ply_mult, c_ply_pow, bonus_val, bonus_ply, max_base, max_ply_mult
manual_params = [0.022, 0.0065, 0.5, 0.06, 12.0, 5.2, 0.01]
manual_error = objective(manual_params)

print(f"Manual Params Error: {manual_error}")

print("\nVerification (Time=60s, Inc=0):")
print(f"{'Ply':<5} {'Orig Opt':<10} {'Simp Opt':<10} {'Ratio':<10} {'Orig Max':<10} {'Simp Max':<10}")
for p in [0, 10, 20, 40, 80, 120]:
    o_opt, o_max = get_time_original(60000, 0, p)
    s_opt, s_max = get_time_simple(manual_params, 60000, 0, p)
    ratio = s_opt / o_opt if o_opt > 0 else 0
    print(f"{p:<5} {o_opt:<10.1f} {s_opt:<10.1f} {ratio:<10.2f} {o_max:<10.1f} {s_max:<10.1f}")
