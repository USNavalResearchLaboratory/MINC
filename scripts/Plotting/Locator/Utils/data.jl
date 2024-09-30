##
function get_loaders_rescaler(run_path::String)
    ##
    cfg = load_object(run_path * "/cfg.jld2")
    ##
    rescaler = MINC.length_units(cfg)
    ##
    rng = cfg.PRNG(cfg.seed)
    loader_train, loader_test = MINC.get_locate_dataloader!(cfg, rng)
    ##
    return loader_train, loader_test, rescaler
end
## READ
function read_data_raw(run_path::String)
    cfg = load_object(run_path * "/cfg.jld2")
    signals_damaged, signals_baseline, locations, times = MINC.read_data(cfg.set)
    return signals_damaged, signals_baseline, locations, times
end
##
function read_data_compressed(run_path::String)
    cfg = load_object(run_path * "/cfg.jld2")
    set = cfg.set
    ω_min = cfg.ω_min
    ω_max = cfg.ω_max
    σ_fknn = cfg.σ_fknn
    σ_rknn = cfg.σ_rknn
    pro_config = (; set, ω_min, ω_max, σ_fknn, σ_rknn)
    ω_modes = ω_min:ω_max
    # Get processed data
    data_dict, file = produce_or_load(
        MINC.get_processed_data, pro_config, datadir("data_pro"); tag=false)
    @unpack signals_damaged_compressed, signals_baseline_compressed, locations = data_dict
    return signals_damaged_compressed, signals_baseline_compressed, locations,
    ω_modes
end
##
function input_target_split(run_path::String)
    #
    loader_train, loader_test, rescaler = get_loaders_rescaler(run_path)
    ##
    target_train = rescaler .* loader_train.data[2]
    target_test = rescaler .* loader_test.data[2]
    input_train = loader_train.data[1]
    input_test = loader_test.data[1]
    ##
    return input_train, target_train, input_test, target_test
end
##
