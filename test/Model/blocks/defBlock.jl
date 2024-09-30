## UTIL FOR TESTS
struct DefBlockTest{F <: Function, S <: String} <: TimeDefToTimeReg
    getModel::F
    name::S
end
##
function DefBlockTest()
    return DefBlockTest(getDefBlock, "DefBlock")
end
##
function getDefBlock(; bs=1, kwargs...)
    ## Config
    cfg = MINC.Config(; epochs=2, infotime=1, batch_size=2, kwargs...)
    ordG = cfg.ordG
    ## Device
    dev = cfg.dev
    ## Channels
    chs = 2
    ## Data
    T = 16
    t = 4
    ## Model
    model = MINC.DefBlock(T, t, ordG, 1 => chs)
    ## Setup
    rng = cfg.PRNG(cfg.seed)
    ps, st = Lux.setup(Lux.replicate(rng), model)
    ## Transfer
    ps = ComponentArray(ps) |> dev
    st = st |> dev
    ## Input
    x = randn(Float32, T, cfg.r, cfg.s, cfg.batch_size * bs) |> dev
    return x, model, ps, st, cfg
end
