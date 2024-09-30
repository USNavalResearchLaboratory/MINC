##
function heatgrid(run_path::String; loader_type=:both)
    ##
    model, ps, st, cfg = get_model_ps_st_cfg(run_path)
    loader_train, loader_test, rescaler = get_loaders_rescaler(run_path)
    ##
    if loader_type == :both
        ## TRAIN
        errors_train, y_train = loader_errors_targets(
            loader_train, model, ps, st, rescaler)
        hgrid_train = make_hgrid(errors_train, y_train)
        ## TEST
        errors_test, y_test = loader_errors_targets(
            loader_test, model, ps, st, rescaler)
        hgrid_test = make_hgrid(errors_test, y_test)
        ##
        return hgrid_train, hgrid_test
    elseif loader_type == :train
        ## TRAIN
        errors_train, y_train = loader_errors_targets(
            loader_train, model, ps, st, rescaler)
        hgrid_train = make_hgrid(errors_train, y_train)
        ##
        return hgrid_train
    elseif loader_type == :test
        ## TEST
        errors_test, y_test = loader_errors_targets(
            loader_test, model, ps, st, rescaler)
        hgrid_test = make_hgrid(errors_test, y_test)
        ##
        return hgrid_test
    end
end
## Utils
function loader_errors_targets(loader, model, ps, st, rescaler)
    ##
    dev = gpu_device()
    ##
    ps = ps |> dev
    st = st |> dev
    ##
    errors_loader = Vector{}()
    _y_loader = Vector{AbstractArray}()
    ##
    for (x, y) in loader
        ##
        y_cpu = y
        x = x |> dev
        y = y |> dev
        ##
        mx, _ = model(x, ps, st)
        ##
        errs = rescaler .*
               map(MINC.md1e, eachslice(mx; dims=2), eachslice(y; dims=2))
        append!(errors_loader, errs)
        append!(_y_loader, eachslice(y_cpu; dims=2))
        ##
    end
    errors_loader = errors_loader |> cpu_device()
    y_loader = stack(_y_loader) |> cpu_device()
    ##
    idx_worst = argmax(errors_loader)
    error_worst = errors_loader[idx_worst]
    y_worst = _y_loader[idx_worst]
    println("Error Worst = $(error_worst)")
    println("Location Worst = $(y_worst)")
    ##
    return errors_loader, y_loader
end
