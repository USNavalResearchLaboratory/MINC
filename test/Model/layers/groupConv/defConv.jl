## UTIL FOR TESTS
struct DefConvTest{F <: Function, S <: String} <: TimeDefToTimeReg
    getModel::F
    name::S
end
##
function DefConvTest()
    return DefConvTest(getDefConv, "DefConv")
end
##
function getDefConv(; bs=1, kwargs...)
    ## Config
    cfg = MINC.Config(; epochs=2, infotime=1, batch_size=2, kwargs...)
    ## Device
    dev = cfg.dev
    ## Defining
    chs_def = 8
    ## Data
    T = 16
    r = cfg.r
    s = cfg.s
    ordG = cfg.ordG
    batch_size = cfg.batch_size
    ## Derived Parameters
    pad = (T - 1, 0)
    ## Model
    model = MINC.DefConv((T,), ordG, 1 => chs_def; pad=pad)
    ## Setup
    rng = cfg.PRNG(cfg.seed)
    ps, st = Lux.setup(Lux.replicate(rng), model)
    ## Transfer
    ps = ComponentArray(ps) |> dev
    st = st |> dev
    ## Input
    x = randn(Float32, T, r, s, batch_size * bs) |> dev
    return x, model, ps, st, cfg
end
