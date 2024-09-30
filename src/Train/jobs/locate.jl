## LOCATE
function locate(; cfg=Config(), dev_id=0, bug_test=false)
    ## GPU
    if CUDA.functional()
        CUDA.allowscalar(false)
        CUDA.device!(dev_id)
        @info "Proceeding on GPU $(dev_id)"
        dev = cfg.dev
    else
        @info "Proceeding on CPU"
        dev = cpu_device()
    end
    # RNG
    rng = cfg.PRNG(cfg.seed)
    ## DATA
    loader_train, loader_test, cfg = get_locate_dataloader!(
        cfg, Lux.replicate(rng); bug_test=bug_test)
    loader = (loader_train, loader_test)
    # MODEL
    model = get_locator(cfg)
    # SETUP
    ps, st = Lux.setup(Lux.replicate(rng), model)
    # TRANSFER
    ps = ComponentArray(ps) |> dev
    st = st |> dev
    ## TRAIN 
    trainer(loader, model, ps, st; cfg=cfg)
    ##
    return nothing
end
