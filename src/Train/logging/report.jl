## REPORT 
function report(loader_train, loader_test, model, ps, st; cfg=Config())
    ##
    epoch = 0
    ##
    metrics_train, metrics_test = get_train_test_metrics(
        epoch, loader_train, loader_test, model, ps, st; cfg=cfg)
    metrics_test_history = Dict(k => [v] for (k, v) in metrics_test)
    metrics_train_history = Dict(k => [v] for (k, v) in metrics_train)
    ## Initial save
    save_metrics(
        epoch, metrics_train_history, metrics_test_history, model; cfg=cfg)
    ##
    return metrics_train_history, metrics_test_history
end
function report(epoch, loader_train, loader_test, metrics_train_history,
        metrics_test_history, model, ps, st; cfg=Config())
    ## *During* loop Metrics 
    if ((epoch % cfg.infotime == 0) && (epoch != cfg.epochs))
        ## Update
        metrics_train_history, metrics_test_history = update_train_test_metrics(
            epoch, loader_train, loader_test, metrics_train_history,
            metrics_test_history, model, ps, st; cfg=cfg)
        ## Save
        save_metrics(
            epoch, metrics_train_history, metrics_test_history, model; cfg=cfg)
        ## *Final* Metrics AND Preds/Params
    elseif epoch == cfg.epochs
        ## Update
        metrics_train_history, metrics_test_history = update_train_test_metrics(
            epoch, loader_train, loader_test, metrics_train_history,
            metrics_test_history, model, ps, st; cfg=cfg)
        ## Save
        save_metrics(
            epoch, metrics_train_history, metrics_test_history, model; cfg=cfg)
        save_ps_st(model, ps, st; cfg=cfg)
        save_preds(loader_test, model, ps, st; cfg=cfg)
        ##
    end
    ##
    return metrics_train_history, metrics_test_history
end
## Utils
function get_train_test_metrics(
        epoch, loader_train, loader_test, model, ps, st; cfg=Config())
    ##
    metrics_train = eval_loss(loader_train, model, ps, st; cfg=cfg)
    metrics_test = eval_loss(loader_test, model, ps, st; cfg=cfg)
    ##
    for key in keys(metrics_train)
        println("Epoch: $(epoch) train $(key): $(metrics_train[key]) test $(key): $(metrics_test[key])")
    end
    return metrics_train, metrics_test
end
##
function update_train_test_metrics(
        epoch, loader_train, loader_test, metrics_train_history,
        metrics_test_history, model, ps, st; cfg=Config())
    ##
    metrics_train, metrics_test = get_train_test_metrics(
        epoch, loader_train, loader_test, model, ps, st; cfg=cfg)
    ## 
    for key in keys(metrics_train)
        metrics_train_history[key] = push!(
            metrics_train_history[key], metrics_train[key])
        metrics_test_history[key] = push!(
            metrics_test_history[key], metrics_test[key])
    end
    return metrics_train_history, metrics_test_history
end
## Save
function save_ps_st(
        model::Union{Detector, Locator, Imager}, ps, st; cfg=Config())
    ps_cpu = ps |> cpu_device()
    st_cpu = st |> cpu_device()
    safesave(projectdir(cfg.savepath * "model/ps_final.jld2"), @strdict ps_cpu)
    safesave(projectdir(cfg.savepath * "model/st_final.jld2"), @strdict st_cpu)
    return nothing
end
function save_ps_st(model, ps, st; cfg=Config())
    # For testing convenience
    # Don't save when passing anything but a Model through trainer
end
