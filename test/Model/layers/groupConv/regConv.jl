## UTIL FOR TESTS
struct RegConvTest{F <: Function, S <: String} <: TimeRegToTimeReg
    getModel::F
    name::S
end
##
function RegConvTest()
    return RegConvTest(getRegConv, "RegConv")
end
##
function getRegConv(; bs=1, kwargs...)
    ## Config
    cfg = MINC.Config(; epochs=2, infotime=1, batch_size=2, kwargs...)
    ## Device
    dev = cfg.dev
    ## Regular
    chs_reg = 8
    ## Data
    T = 16
    r = cfg.r
    s = cfg.s
    ordG = cfg.ordG
    batch_size = cfg.batch_size
    ## Derived Parameters
    pad = (T - 1, 0)
    ## Model
    model = MINC.RegConv((T,), ordG, chs_reg => chs_reg; pad=pad)
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
