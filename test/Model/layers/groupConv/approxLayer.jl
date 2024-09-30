## UTIL FOR TESTS
struct ApproxLayerTest{F <: Function, S <: String} <: RToR
    getModel::F
    name::S
end
##
function ApproxLayerTest()
    return ApproxLayerTest(getApproxLayer, "ApproxLayer")
end
##
function getApproxLayer(; bs=1, kwargs...)
    ## Config
    cfg = MINC.Config(; epochs=2, infotime=1, batch_size=2, kwargs...)
    ## Device
    dev = cfg.dev
    ## Regular
    chs_reg = 2
    ## Data
    T = 16
    r = cfg.r
    s = cfg.s
    ordG = cfg.ordG
    batch_size = cfg.batch_size
    ## Derived Parameters
    pad = (T - 1, 0)
    ## Model
    model = MINC.ApproxLayer{Val(0)}(ordG)
    ## Setup
    rng = cfg.PRNG(cfg.seed)
    ps, st = Lux.setup(Lux.replicate(rng), model)
    ## Transfer
    ps = ComponentArray(ps) |> dev
    st = st |> dev
    ## Input
    x = randn(Float32, T, ordG, chs_reg, batch_size * bs) |> dev
    return x, model, ps, st, cfg
end
