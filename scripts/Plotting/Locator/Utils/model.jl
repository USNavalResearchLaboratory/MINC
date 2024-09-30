## GET
function get_saved_array(run_path::String, observable::String)
    array_paths = readdir(run_path * "/arrays"; join=true)
    observable_path = only(filter(s -> occursin(observable, s), array_paths))
    return load_object(observable_path)
end
function get_final_model(run_path, observable::String)
    ##
    model_paths = readdir(run_path * "/model"; join=true)
    observable_path = only(filter(s -> occursin(observable, s), model_paths))
    ##
    return load_object(observable_path)
end
##
function get_model_ps_st_cfg(run_path::String)
    ##
    cfg = load_object(run_path * "/cfg.jld2")
    cfg.partial = true
    cfg.dev = gpu_device()
    model = MINC.get_locator(cfg)
    #  xxx the following doesn't always load nicely
    #  so we remake it above 
    #model = load_object(run_path * "/model/model.jld2")
    ##
    ps = get_final_model(run_path, "ps")
    st = get_final_model(run_path, "st")
    ##
    return model, ps, st, cfg
end
##
