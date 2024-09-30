## FORWARD
function test_forward(ModelTest)
    x, m, ps, st, cfg = ModelTest.getModel()
    @info "Checking $(ModelTest.name) forward"
    m_x, st_ = m(x, ps, st)
    return m_x
end
## BACKWARD
function test_backward(ModelTest)
    x, m, ps, st, cfg = ModelTest.getModel()
    @info "Checking $(ModelTest.name) backward"
    m_x, st_ = m(x, ps, st)
    dev = cfg.dev
    l, gs = _test_backward(x, m_x, m, ps, st, dev, ModelTest)
    return l, gs
end
function _test_backward(x, m_x, m, ps, st, dev, ModelTest)
    y = randn(eltype(m_x), size(m_x)) |> dev
    (l, st_), back = Zygote.pullback(p -> MINC.Loss_st(x, y, m, p, st), ps)
    gs = back((one(l), nothing))[1]
    return l, gs
end
function _test_backward(x, m_x, m, ps, st, dev,
        ModelTest::Union{DetectorTest, ToTwoScalarsTest})
    ##
    y = abs.(randn(eltype(m_x), size(m_x))) |> dev
    y = y ./ maximum(y)
    ##
    (l, st_), back = Zygote.pullback(p -> MINC.Loss_st(x, y, m, p, st), ps)
    gs = back((one(l), nothing))[1]
    return l, gs
end
## EQUIVARIANCE
function test_equivariance(ModelTest)
    x, m, ps, st, cfg = ModelTest.getModel()
    @info "Checking $(ModelTest.name) equivariance"
    m_x, st_ = m(x, ps, st)
    areapprox, maxerr = _test_equivariance(x, m_x, m, ps, st, cfg, ModelTest)
    println((areapprox=areapprox, maxerr=maxerr))
    return areapprox
end
function _test_equivariance(x, m_x, m, ps, st, cfg, ModelTest)
    return false, 0
end
function _test_equivariance(x, m_x, m, ps, st, cfg, ModelTest::TimeDefToGrid)
    ##
    G_domain_indices = get_G_timedef_indices(x; cfg=cfg)
    G_codomain_indices = get_G_grid_indices(m_x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices, G_codomain_indices; cfg=cfg)
    ##
    return areapprox, maxerr
end
function _test_equivariance(x, m_x, m, ps, st, cfg, ModelTest::TimeRegToGrid)
    ##
    G_domain_indices = get_G_timereg_indices(x; cfg=cfg)
    G_codomain_indices = get_G_grid_indices(m_x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices, G_codomain_indices; cfg=cfg)
    ##
    return areapprox, maxerr
end
function _test_equivariance(x, m_x, m, ps, st, cfg, ModelTest::TimeDefToTimeReg)
    ##
    G_domain_indices = get_G_timedef_indices(x; cfg=cfg)
    G_codomain_indices = get_G_timereg_indices(m_x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices, G_codomain_indices; cfg=cfg)
    return areapprox, maxerr
end
function _test_equivariance(x, m_x, m, ps, st, cfg, ModelTest::TimeRegToTimeReg)
    ##
    G_domain_indices = get_G_timereg_indices(x; cfg=cfg)
    G_codomain_indices = get_G_timereg_indices(m_x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices, G_codomain_indices; cfg=cfg)
    return areapprox, maxerr
end
function _test_equivariance(x, m_x, m, ps, st, cfg, ModelTest::TimeDefToVec)
    ##
    G_domain_indices = get_G_timedef_indices(x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices; cfg=cfg)
    return areapprox, maxerr
end
function _test_equivariance(x, m_x, m, ps, st, cfg, ModelTest::TimeRegToVec)
    ##
    G_domain_indices = get_G_timereg_indices(x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices; cfg=cfg)
    return areapprox, maxerr
end
function _test_equivariance(x, m_x, m, ps, st, cfg, ModelTest::TimeDefToBin)
    ##
    G_domain_indices = get_G_timedef_indices(x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices; cfg=cfg)
    return areapprox, maxerr
end
function _test_equivariance(x, m_x, m, ps, st, cfg, ModelTest::TimeRegToBin)
    ##
    G_domain_indices = get_G_timereg_indices(x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices; cfg=cfg)
    return areapprox, maxerr
end
## TRAINER
function test_trainer(ModelTest; bs=5)
    ##
    x, m, ps, st, cfg = ModelTest.getModel(; bs=bs)
    @info "Checking $(ModelTest.name) trainer"
    # Device
    dev = cfg.dev
    m_x, st_ = m(x, ps, st)
    ##
    loader = _test_trainer(x, m_x, m, ps, st, dev, cfg, ModelTest; bs=bs)
    ##
    MINC.trainer(loader, m, ps, st; cfg=cfg)
    return nothing
end
function _test_trainer(x, m_x, m, ps, st, dev, cfg, ModelTest; bs=5)
    y = randn(eltype(m_x), size(m_x)) |> dev
    D = (x |> cpu_device(), y |> cpu_device())
    rng = cfg.PRNG(cfg.seed)
    loader = get_dataloader(Lux.replicate(rng), D; cfg)
    return loader
end
function _test_trainer(x, m_x, m, ps, st, dev, cfg,
        ModelTest::Union{DetectorTest, ToTwoScalarsTest}; bs=5)
    y = abs.(randn(eltype(m_x), size(m_x))) |> dev
    y = y ./ maximum(y)
    D = (x |> cpu_device(), y |> cpu_device())
    rng = cfg.PRNG(cfg.seed)
    loader = get_dataloader(Lux.replicate(rng), D; cfg)
    return loader
end
## TRAINED EQUIVARIANCE
function test_trainedequivariance(ModelTest::TimeDefToGrid; bs=5)
    ##
    x, m, ps, st, cfg = ModelTest.getModel(; bs=bs)
    @info "Checking $(ModelTest.name) trained equivariance"
    ## Device
    dev = cfg.dev
    m_x, st_ = m(x, ps, st)
    ##
    y = randn(eltype(m_x), size(m_x)) |> dev
    D = (x |> cpu_device(), y |> cpu_device())
    rng = cfg.PRNG(cfg.seed)
    loader = get_dataloader(Lux.replicate(rng), D; cfg)
    MINC.trainer(loader, m, ps, st; cfg=cfg)
    m_x, st_ = m(x, ps, st)
    ##
    G_domain_indices = get_G_timedef_indices(x; cfg=cfg)
    G_codomain_indices = get_G_grid_indices(m_x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices, G_codomain_indices; cfg=cfg)
    ##
    println((areapprox=areapprox, maxerr=maxerr))
    return areapprox
end
function test_trainedequivariance(ModelTest::TimeRegToGrid; bs=5)
    ##
    x, m, ps, st, cfg = ModelTest.getModel(; bs=bs)
    @info "Checking $(ModelTest.name) trained equivariance"
    ## Device
    dev = cfg.dev
    m_x, st_ = m(x, ps, st)
    ##
    y = randn(eltype(m_x), size(m_x)) |> dev
    D = (x |> cpu_device(), y |> cpu_device())
    rng = cfg.PRNG(cfg.seed)
    loader = get_dataloader(Lux.replicate(rng), D; cfg)
    MINC.trainer(loader, m, ps, st; cfg=cfg)
    m_x, st_ = m(x, ps, st)
    ##
    G_domain_indices = get_G_timereg_indices(x; cfg=cfg)
    G_codomain_indices = get_G_grid_indices(m_x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices, G_codomain_indices; cfg=cfg)
    ##
    println((areapprox=areapprox, maxerr=maxerr))
    return areapprox
end
function test_trainedequivariance(ModelTest::TimeDefToTimeReg; bs=5)
    ##
    x, m, ps, st, cfg = ModelTest.getModel(; bs=bs)
    @info "Checking $(ModelTest.name) trained equivariance"
    ## Device
    dev = cfg.dev
    m_x, st_ = m(x, ps, st)
    ##
    y = randn(eltype(m_x), size(m_x)) |> dev
    D = (x |> cpu_device(), y |> cpu_device())
    rng = cfg.PRNG(cfg.seed)
    loader = get_dataloader(Lux.replicate(rng), D; cfg)
    MINC.trainer(loader, m, ps, st; cfg=cfg)
    m_x, st_ = m(x, ps, st)
    ##
    G_domain_indices = get_G_timedef_indices(x; cfg=cfg)
    G_codomain_indices = get_G_timereg_indices(m_x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices, G_codomain_indices; cfg=cfg)
    ##
    println((areapprox=areapprox, maxerr=maxerr))
    return areapprox
end
##
function test_trainedequivariance(ModelTest::TimeRegToTimeReg; bs=5)
    ##
    x, m, ps, st, cfg = ModelTest.getModel(; bs=bs)
    @info "Checking $(ModelTest.name) trained equivariance"
    ## Device
    dev = cfg.dev
    m_x, st_ = m(x, ps, st)
    ##
    y = randn(eltype(m_x), size(m_x)) |> dev
    D = (x |> cpu_device(), y |> cpu_device())
    rng = cfg.PRNG(cfg.seed)
    loader = get_dataloader(Lux.replicate(rng), D; cfg)
    MINC.trainer(loader, m, ps, st; cfg=cfg)
    m_x, st_ = m(x, ps, st)
    ##
    G_domain_indices = get_G_timereg_indices(x; cfg=cfg)
    G_codomain_indices = get_G_timereg_indices(m_x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices, G_codomain_indices; cfg=cfg)
    ##
    println((areapprox=areapprox, maxerr=maxerr))
    return areapprox
end
##
function test_trainedequivariance(ModelTest::TimeDefToVec; bs=5)
    ##
    x, m, ps, st, cfg = ModelTest.getModel(; bs=bs)
    @info "Checking $(ModelTest.name) trained equivariance"
    ## Device
    dev = cfg.dev
    ##
    m_x, st_ = m(x, ps, st)
    y = randn(eltype(m_x), size(m_x)) |> dev
    D = (x |> cpu_device(), y |> cpu_device())
    rng = cfg.PRNG(cfg.seed)
    loader = get_dataloader(Lux.replicate(rng), D; cfg)
    MINC.trainer(loader, m, ps, st; cfg=cfg)
    m_x, st_ = m(x, ps, st)
    ##
    G_domain_indices = get_G_timedef_indices(x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices; cfg=cfg)
    ##
    println((areapprox=areapprox, maxerr=maxerr))
    return areapprox
end
function test_trainedequivariance(ModelTest::TimeRegToVec; bs=5)
    ##
    x, m, ps, st, cfg = ModelTest.getModel(; bs=bs)
    @info "Checking $(ModelTest.name) trained equivariance"
    ## Device
    dev = cfg.dev
    m_x, st_ = m(x, ps, st)
    y = randn(eltype(m_x), size(m_x)) |> dev
    D = (x |> cpu_device(), y |> cpu_device())
    rng = cfg.PRNG(cfg.seed)
    loader = get_dataloader(Lux.replicate(rng), D; cfg)
    MINC.trainer(loader, m, ps, st; cfg=cfg)
    m_x, st_ = m(x, ps, st)
    ##
    G_domain_indices = get_G_timereg_indices(x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices; cfg=cfg)
    ##
    println((areapprox=areapprox, maxerr=maxerr))
    return areapprox
end
function test_trainedequivariance(ModelTest::TimeDefToBin; bs=5)
    ##
    x, m, ps, st, cfg = ModelTest.getModel(; bs=bs)
    @info "Checking $(ModelTest.name) trained equivariance"
    ## Device
    dev = cfg.dev
    ##
    m_x, st_ = m(x, ps, st)
    y = abs.(randn(eltype(m_x), size(m_x))) |> dev
    y = y ./ maximum(y)
    D = (x |> cpu_device(), y |> cpu_device())
    rng = cfg.PRNG(cfg.seed)
    loader = get_dataloader(Lux.replicate(rng), D; cfg)
    MINC.trainer(loader, m, ps, st; cfg=cfg)
    m_x, st_ = m(x, ps, st)
    ##
    G_domain_indices = get_G_timedef_indices(x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices; cfg=cfg)
    ##
    println((areapprox=areapprox, maxerr=maxerr))
    return areapprox
end
function test_trainedequivariance(ModelTest::TimeRegToBin; bs=5)
    ##
    x, m, ps, st, cfg = ModelTest.getModel(; bs=bs)
    @info "Checking $(ModelTest.name) trained equivariance"
    ## Device
    dev = cfg.dev
    ##
    m_x, st_ = m(x, ps, st)
    ##
    y = abs.(randn(eltype(m_x), size(m_x))) |> dev
    y = y ./ maximum(y)
    D = (x |> cpu_device(), y |> cpu_device())
    rng = cfg.PRNG(cfg.seed)
    loader = get_dataloader(Lux.replicate(rng), D; cfg)
    m_x, st_ = m(x, ps, st)
    ##
    G_domain_indices = get_G_timereg_indices(x; cfg=cfg)
    ##
    areapprox, maxerr = equivarianceError(
        x, m, ps, st, m_x, G_domain_indices; cfg=cfg)
    ##
    println((areapprox=areapprox, maxerr=maxerr))
    return areapprox
end
## MASTER TEST
function test_model(ModelTest::RepToRep)
    test_forward(ModelTest)
    test_backward(ModelTest)
    wasapprox = test_equivariance(ModelTest)
    test_trainer(ModelTest)
    areapprox = test_trainedequivariance(ModelTest)
    return (wasapprox || areapprox)
end
function test_model(ModelTest::RToR)
    test_forward(ModelTest)
    test_backward(ModelTest)
    test_trainer(ModelTest)
    return true
end
## EQUIVARIANCE ERROR
function equivarianceError(x, m, ps, st, m_x, G_domain_indices,
        G_codomain_indices; cfg=MINC.Config())
    G_x = map(g_domain_ind -> x[g_domain_ind], G_domain_indices)
    es_m_G_x = map(g_x -> first(m(g_x, ps, st)), G_x)
    ##
    G_m_x = map(g_codomain_ind -> m_x[g_codomain_ind], G_codomain_indices)
    ##
    er = map(gmx_mgx -> norm(gmx_mgx), G_m_x .- es_m_G_x)
    tf = map(isapprox, G_m_x, es_m_G_x)
    ##
    areapprox = ~(0 in tf)
    maxerr = maximum(er)
    ##
    return areapprox, maxerr
end
function equivarianceError(x, m::Union{MINC.Locator, MINC.ToVector}, ps,
        st, m_x, G_domain_indices; cfg=MINC.Config())
    ##
    G_x = map(g_domain_ind -> x[g_domain_ind], G_domain_indices)
    es_m_G_x = map(g_x -> first(m(g_x, ps, st)), G_x)
    ##
    es_m_G_x = es_m_G_x |> cpu_device()
    m_x = m_x |> cpu_device()
    R = MINC.get_vec_rep(cfg.ordG) |> cpu_device()
    ## xxx this slicemap doesn't do what I want it to on the gpu (???)
    G_m_x = map(g -> g * m_x, eachslice(R; dims=1))
    ##
    er = map(gmx_mgx -> norm(gmx_mgx), G_m_x .- es_m_G_x)
    tf = map(isapprox, G_m_x, es_m_G_x)
    ##
    areapprox = ~(0 in tf)
    maxerr = maximum(er)
    ##
    return areapprox, maxerr
end
function equivarianceError(x, m::Union{MINC.Detector, MINC.ToTwoScalars},
        ps, st, m_x, G_domain_indices; cfg=MINC.Config())
    ##
    G_x = map(g_domain_ind -> x[g_domain_ind], G_domain_indices)
    es_m_G_x = map(g_x -> first(m(g_x, ps, st)), G_x)
    ##
    G_m_x = map(g -> m_x, 1:(cfg.ordG))
    ##
    er = map(gmx_mgx -> norm(gmx_mgx), G_m_x .- es_m_G_x)
    tf = map(isapprox, G_m_x, es_m_G_x)
    ##
    areapprox = ~(0 in tf)
    maxerr = maximum(er)
    ##
    return areapprox, maxerr
end
## DISPATCH FOR GETTING INDICES
function get_G_grid_indices(x::AbstractArray{Float32, 3}; cfg=Config()) #size(x) ~ [x, y, b]
    ##
    grid_length = cfg.grid_length
    ## [i, x, y, g]
    grid_rep = MINC.get_grid_rep(grid_length)
    # [i, x * y * g]
    grid_rep_r = reshape(grid_rep, 2, :)
    # [x * y * g][i]
    grid_inds_r = _eachslice(grid_rep_r, Val(2))
    ##
    cart_indices = CartesianIndices(x)
    ##
    G_cart_indices_r = stack(
        map(inds -> cart_indices[inds..., :], grid_inds_r); dims=1)
    ##
    G_cart_indices = _eachslice(
        reshape(G_cart_indices_r, grid_length, grid_length, cfg.ordG, :),
        Val(3))
    ##
    return G_cart_indices
end
function get_G_timedef_indices(x::AbstractArray{Float32, 4}; cfg=Config()) #size(x) ~ [t, r, s, b]
    ##
    def_indices = _eachslice(MINC.get_defining_perms(cfg.ordG), Val(2))
    cart_indices = CartesianIndices(x)
    G_cart_indices = map(inds -> cart_indices[:, inds, inds, :], def_indices)
    ##
    return G_cart_indices
end
##
function get_G_timereg_indices(x::AbstractArray{Float32, 4}; cfg=Config()) #size(x) ~ [t, g, c, b]
    ##
    reg_indices = _eachslice(MINC.get_regular_perms(cfg.ordG), Val(2))
    cart_indices = CartesianIndices(x)
    G_cart_indices = map(inds -> cart_indices[:, inds, :, :], reg_indices)
    ##
    return G_cart_indices
end
##
