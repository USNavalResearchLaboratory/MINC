##
function length_units(cfg::Config)
    ##
    L_max = cfg.L_max
    L_min = cfg.L_min
    rescaler = (L_max - L_min) / 2
    ##
    return rescaler
end
##
function time_compress(t, cfg::Config)
    ##
    ω_min = cfg.ω_min
    ω_max = cfg.ω_max
    ω_min_compressed, ω_max_compressed = freq_compress(ω_min, ω_max)
    ω_modes = ω_min_compressed:ω_max_compressed
    ##
    Γ = 2 * (length(ω_modes) - 1) # window compress to Γ units
    T = 0.4 # 0.4 ms measurement window
    t_compressed_step = round(Int, t * (Γ / T))
    return t_compressed_step
end
function time_compress(t_start, t_end, cfg::Config)
    t_start_compressed = time_compress(t_start, cfg)
    t_end_compressed = time_compress(t_end, cfg)
    return t_start_compressed, t_end_compressed
end
function time_compress(cfg::Config)
    t_start = cfg.t_start
    t_end = cfg.t_end
    return time_compress(t_start, t_end, cfg)
end
##
function freq_compress(ω_min, ω_max)
    ## xxx read_data gets times but i prefer the interval to be [0, 0.4]
    times = range(0, 0.4f-3, 10_000)
    dt = mean(diff(times))
    freqs = rfftfreq(length(times), 1 / dt) ./ 10^3 #kHz
    #
    ω_min_compressed = maximum(findall(freqs .< ω_min))
    ω_max_compressed = minimum(findall(freqs .> ω_max))
    ##
    return ω_min_compressed, ω_max_compressed
end
##
function get_T_over_gamma(cfg::Config)
    ##
    ω_min = cfg.ω_min
    ω_max = cfg.ω_max
    ω_min_compressed, ω_max_compressed = freq_compress(ω_min, ω_max)
    ω_modes = ω_min_compressed:ω_max_compressed
    ##
    Γ = 2 * (length(ω_modes) - 1) # window compress to Γ units
    ## xxx read_data gets times but i prefer the interval to be [0, 0.4]
    T = 0.4 # 0.4 ms measurement window
    return T / Γ
end
