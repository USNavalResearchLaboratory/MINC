##
function prepare_locate_data(cfg::Config)
    ##
    signals_damaged_compressed, signals_baseline_compressed, locations = MINC.get_processed_data(cfg)
    ## Prepare data config
    subtract_baseline = cfg.subtract_baseline
    zero_diagonals = cfg.zero_diagonals
    t_start = cfg.t_start
    t_end = cfg.t_end
    σ_maxabs = cfg.σ_maxabs
    ##
    """
    If subtract_baseline,
        size(signals_damaged_blsub) = (t, r, s, bl, b)
        size(locations_blsub) = (i, bl, b)
    Otherwise,
        size(signals_damaged_blsub) = (t, r, s, b)
        size(locations_blsub) = (i, b)
    """
    ##
    signals_damaged_blsub, locations_blsub = baseline_subtract(
        subtract_baseline, signals_damaged_compressed,
        signals_baseline_compressed, locations)
    ##
    """
    Provided locations are not measured about the plate center.
    The following shifts the origin to the center of the plate
        and normalizes the location components to lie 
        in the interval [-1, 1]
    """
    ## Normalize/shift target 
    L_max = maximum(locations_blsub)
    L_min = minimum(locations_blsub)
    L_shift = Float32((L_max + L_min) / 2)
    L_rescale = L_max .- L_shift
    target_data = (locations_blsub .- L_shift) ./ L_rescale
    ##
    """
    Since all send waveform are meant to be identical, it is thought not necessary
        to retain their structure in the training data.
    Also, the diagonals (sends) hold noise that is on the order of the off-diagonals (receives);
        this noise is a function of the transducer transients and does not contain information 
        reflecting wave propagation through damage.
    Hence, we set the diagonals to zero.
    """
    ## Zero Diagonals
    if zero_diagonals
        signals_damaged_blsub_zd = zero_diagonals!(signals_damaged_blsub)
    else
        signals_damaged_blsub_zd = signals_damaged_blsub
    end
    """
    After baseline subtraction, a few examples develop anomalous features
    """
    ## Discard more pathological examples
    if subtract_baseline
        signals_damaged_blsub_zd_pp, target_data_pp = post_process_locate_data(
            signals_damaged_blsub_zd, target_data; σ_maxabs=σ_maxabs)
    end
    ##
    """
    All times before the signal arrives at the receiver are irrelevant, meaning
        we can look only at times t ≥ t_start. Can further choose t_end to eliminate
        reflections.
    """
    ##
    signals_damaged_blsub_zd_pp_ws = window_signals(
        signals_damaged_blsub_zd_pp, t_start, t_end; cfg=cfg)
    ## Normalize input
    sup = maximum(abs.(signals_damaged_blsub_zd_pp_ws))
    input_data = signals_damaged_blsub_zd_pp_ws ./ (sup + 1.0f-9)
    ##
    return input_data, target_data_pp, L_max, L_min
end
