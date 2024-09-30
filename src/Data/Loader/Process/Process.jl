# PROCESS DATA
include("fknn.jl")
include("rknn.jl")
##
function get_processed_data(cfg::Config)
    ## Process data config
    set = cfg.set
    ω_min = cfg.ω_min
    ω_max = cfg.ω_max
    σ_fknn = cfg.σ_fknn
    σ_rknn = cfg.σ_rknn
    pro_config = (; set, ω_min, ω_max, σ_fknn, σ_rknn)
    ## Get processed data
    data_dict, file = produce_or_load(
        get_processed_data, pro_config, datadir("data_pro"); tag=false)
    @unpack signals_damaged_compressed, signals_baseline_compressed, locations = data_dict
    ##
    return signals_damaged_compressed, signals_baseline_compressed, locations
end
##
function get_processed_data(pro_config::NamedTuple)
    ##
    @unpack set, ω_min, ω_max, σ_fknn, σ_rknn = pro_config
    ω_min_compressed, ω_max_compressed = freq_compress(ω_min, ω_max)
    ω_modes = ω_min_compressed:ω_max_compressed
    Γ = 2 * (length(ω_modes) - 1)
    ## Read
    signals_damaged, signals_baseline, locations = read_data(set)
    ## Zero Mean
    signals_damaged = map(signal -> signal .- mean(signal), signals_damaged)
    signals_baseline = map(signal -> signal .- mean(signal), signals_baseline)
    ## High/Low pass
    signals_damaged_trafod = map(
        signal -> copy(view(rfft(signal), ω_modes)), signals_damaged)
    signals_baseline_trafod = map(
        signal -> copy(view(rfft(signal), ω_modes)), signals_baseline)
    ## Discard spectral knn-outliers
    signals_damaged_trafod, signals_baseline_trafod, locations = discard_spectral_outliers(
        signals_damaged_trafod, signals_baseline_trafod,
        locations; σ_fknn=σ_fknn)
    ## Back trafo
    signals_damaged_compressed = map(
        signal_trafod -> irfft(signal_trafod, Γ), signals_damaged_trafod)
    signals_baseline_compressed = map(
        signal_trafod -> irfft(signal_trafod, Γ), signals_baseline_trafod)
    ## Zero Mean
    signals_damaged_compressed = map(
        signal -> signal .- mean(signal), signals_damaged_compressed)
    signals_baseline_compressed = map(
        signal -> signal .- mean(signal), signals_baseline_compressed)
    ## Discard real knn-outliers
    signals_damaged_compressed, signals_baseline_compressed, locations = discard_time_outliers(
        signals_damaged_compressed,
        signals_baseline_compressed, locations; σ_rknn=σ_rknn)
    ## signals have shape [r, s, batch][t] 
    ## locations has size [batch][i]
    return @strdict signals_damaged_compressed signals_baseline_compressed locations
end
