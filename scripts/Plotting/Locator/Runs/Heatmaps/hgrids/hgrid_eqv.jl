##
function heatgrid_eqv(run_path::String, g::Int; loader_type=:both)
    ##
    model, ps, st, cfg = get_model_ps_st_cfg(run_path)
    #
    loader_train, loader_test, rescaler = get_loaders_rescaler(run_path)
    ##
    if loader_type == :both
        ## TRAIN
        errors_train, y_train = loader_eqv_errors_targets(
            loader_train, model, ps, st, rescaler, g)
        hgrid_train = make_hgrid(errors_train, y_train)
        ## TEST
        errors_test, y_test = loader_eqv_errors_targets(
            loader_test, model, ps, st, rescaler, g)
        hgrid_test = make_hgrid(errors_test, y_test)
        ##
        return hgrid_train, hgrid_test
    elseif loader_type == :train
        ## TRAIN
        errors_train, y_train = loader_eqv_errors_targets(
            loader_train, model, ps, st, rescaler, g)
        hgrid_train = make_hgrid(errors_train, y_train)
        return hgrid_train
    elseif loader_type == :test
        ## TEST
        errors_test, y_test = loader_eqv_errors_targets(
            loader_test, model, ps, st, rescaler, g)
        hgrid_test = make_hgrid(errors_test, y_test)
        ##
        return hgrid_test
    end
end
## Utils
function loader_eqv_errors_targets(loader, model, ps, st, rescaler, g)
    ##
    dev = gpu_device()
    ##
    ps = ps |> dev
    st = st |> dev
    ##
    ρ_vec = MINC.get_vec_rep(8)
    ρ_def = MINC.get_defining_perms(8)
    ##
    g_def = ρ_def[:, g]
    g_vec = ρ_vec[g, :, :] |> dev
    ##
    errors_g = Vector{}()
    _y_loader = Vector{AbstractArray}()
    ##
    for (x, y) in loader
        ##
        y_cpu = y
        x = x |> dev
        y = y |> dev
        ##
        mx, _ = model(x, ps, st)
        gmx = apply_g_to_y(mx, g_vec)
        gx = apply_g_to_x(x, g_def)
        mgx, _ = model(gx, ps, st)
        ##
        errs = rescaler .*
               map(MINC.md1e, eachslice(gmx; dims=2), eachslice(mgx; dims=2))
        ##
        append!(errors_g, errs)
        append!(_y_loader, eachslice(y_cpu; dims=2))
        ##
    end
    errors_g = errors_g |> cpu_device()
    y_loader = stack(_y_loader) |> cpu_device()
    return errors_g, y_loader
end
##
