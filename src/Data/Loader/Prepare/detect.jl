## Detect
function prepare_detect_data(cfg::Config)
    ##
    signals_damaged_compressed, signals_baseline_compressed, locations = get_processed_data(cfg)
    ## Prepare data config
    zero_diagonals = cfg.zero_diagonals
    t_start = cfg.t_start
    t_end = cfg.t_end
    class_ratio = cfg.class_ratio
    ##
    signals_baseline = balance_classes_maybe(
        class_ratio, signals_damaged_compressed, signals_baseline_compressed)
    signals_damaged = stack(signals_damaged_compressed)
    ##
    loaded_or_not = get_loaded_or_not(signals_damaged, signals_baseline)
    ##
    signals = cat(
        signals_damaged, signals_baseline; dims=ndims(signals_damaged))
    target_data = stack(stack(loaded_or_not))
    ##
    """
    Since all send waveform are meant to be identical, it is thought not necessary
        to retain their structure in the training data.
    Also, the diagonals (sends) hold noise that is on the order of the off-diagonals (receives);
        this noise is a function of the transducer transients and does not contain information 
        reflecting wave propagation through damage.
    Hence, we set the diagonals to zero.
    """
    if zero_diagonals
        signals_zd = zero_diagonals!(signals)
    end
    """
    All times before the signal arrives at the receiver are irrelevant, meaning
        we can look only at times t ≥ t_start. Can also choose t_end to eliminate 
        reflections.
    """
    signals_zd_ws = window_signals(signals_zd, t_start, t_end; cfg=cfg)
    ## Normalize input
    sup = maximum(abs.(signals_zd_ws))
    input_data = signals_zd_ws ./ (sup + 1.0f-9)
    ##
    return input_data, target_data
end
##
function get_loaded_or_not(
        signals_damaged::Array{Float32, 4}, signals_baseline::Array{Float32, 4})
    ## size(signals) = (t, r, s, b)
    N_damaged = size(signals_damaged, 4)
    N_baseline_new = size(signals_baseline, 4)
    ##
    loaded = map(i -> onehot(true, [true, false]), 1:N_damaged)
    not_loaded = map(i -> onehot(false, [true, false]), 1:N_baseline_new)
    loaded_or_not = vcat(loaded, not_loaded)
    ##
    return loaded_or_not
end
function balance_classes_maybe(class_ratio, signals_damaged, signals_baseline)
    if class_ratio > 0
        ##
        N_damaged = size(signals_damaged, ndims(signals_damaged))
        N_baseline = size(signals_baseline, ndims(signals_baseline))
        ##
        N_baseline_new = Int64(round(class_ratio * N_damaged))
        coeffs = softmax(randn(Float32, N_baseline_new, N_baseline); dims=2)
        ##
        _signals_baseline = stack(signals_baseline)
        __signals_baseline_mixed = map(sbc -> coeffs * sbc,
            eachslice(_signals_baseline;
                dims=Tuple(1:(ndims(_signals_baseline) - 1))))
        _signals_baseline_mixed = stack(eachslice(
            stack(__signals_baseline_mixed); dims=1))
        ##
    else
        _signals_baseline_mixed = stack(signals_baseline)
    end
    return _signals_baseline_mixed
end
