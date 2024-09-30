##
include("Locate/Locate.jl")
include("Detect/Detect.jl")
##
function get_cfg_exp(ratio; exp_type=:Full)
    # For experiments other than full, we increase the sampled
    # frequencies in order to keep the sequence length unchanged
    if exp_type == :Full
        t_start = 0.0725
        t_end = 0.4
        ω_min = 180
        ω_max = 419
    elseif exp_type == :Truncate
        t_start = 0.0725
        t_end = 0.2395
        ω_min = 68
        ω_max = 538
    elseif exp_type == :Skip
        t_start = 0.16
        t_end = 0.4
        ω_min = 137
        ω_max = 462
    elseif exp_type == :Window
        t_start = 0.16
        t_end = 0.2395
        ω_min = 1
        ω_max = 980
    end
    cfg_exp = (; ratio, t_start, t_end, ω_min, ω_max)
    return cfg_exp
end
##
function get_save_name_exp(cfg_exp::NamedTuple)
    @unpack ratio, t_start, t_end, ω_min, ω_max = cfg_exp
    ratio = round(Int, ratio * 100)
    t0 = round(Int, t_start * 10_000)
    t1 = round(Int, t_end * 10_000)
    ω0 = round(Int, ω_min)
    ω1 = round(Int, ω_max)
    save_name = "ratio=$(ratio)_t0=$(t0)_t1=$(t1)_w0=$(ω0)_w1=$(ω1)"
    return save_name
end
