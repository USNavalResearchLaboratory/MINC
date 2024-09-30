## TRAINING LOOP
function trainer(
        loader, model, ps::ComponentArray, st::NamedTuple; cfg=Config())
    ## Unpack
    dev = cfg.dev
    # Dataloader
    loader_train, loader_test = loader
    ##
    @info "Train Examples: $(numobs(loader_train.data))"
    @info "Test Examples: $(numobs(loader_test.data))"
    @info "Model Parameters: $(Lux.parameterlength(model))"
    ## Optimizer/Schedule
    opt = OptimiserChain(
        ClipGrad(cfg.δ), cfg.Optimiser(cfg.η_start), WeightDecay(cfg.λ))
    schedule = OneCycle(cfg.epochs, cfg.η; startval=cfg.η_start,
        endval=cfg.η_end, percent_start=cfg.percent_start)
    ## Initial save/report
    safesave(projectdir(cfg.savepath * "cfg.jld2"), @strdict cfg)
    safesave(projectdir(cfg.savepath * "model/model.jld2"), @strdict model)
    metrics_train_history, metrics_test_history = report(
        loader_train, loader_test, model, ps, Lux.testmode(st); cfg=cfg)
    ## Train States
    st_opt = Optimisers.setup(opt, ps)
    st = Lux.trainmode(st)
    ##
    @info "Begin Training"
    for epoch in 1:(cfg.epochs)
        ## LR Update
        st_opt = Optimisers.adjust(st_opt, schedule(epoch))
        ## Train Step
        ps, st, st_opt = train_step(loader_train, model, ps, st, st_opt, dev)
        ## Save/report
        metrics_train_history, metrics_test_history = report(
            epoch, loader_train, loader_test, metrics_train_history,
            metrics_test_history, model, ps, Lux.testmode(st); cfg=cfg)
        ##
    end
    return nothing
end
##
function train_step(loader_train, model, ps, st, st_opt, dev)
    @showprogress for (x, y) in loader_train
        ##
        x = x |> dev
        y = y |> dev
        ##
        (l, st), back = Zygote.pullback(p -> Loss_st(x, y, model, p, st), ps)
        # We need to add `nothing`s equal to the number of returned values - 1
        gs = back((one(l), nothing))[1]
        st_opt, ps = Optimisers.update!(st_opt, ps, gs)
    end
    return ps, st, st_opt
end
