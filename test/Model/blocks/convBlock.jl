## UTIL FOR TESTS
struct ConvBlockTest{F <: Function, S <: String} <: RToR
    getModel::F
    name::S
end
##
function ConvBlockTest()
    return ConvBlockTest(getConvBlock, "ConvBlock")
end
##
function getConvBlock(; bs=1, kwargs...)
    ## Config
    cfg = MINC.Config(; epochs=2, infotime=1, batch_size=2, kwargs...)
    ordG = cfg.ordG
    ## Device
    dev = cfg.dev
    ## Channels
    chs = cfg.r * cfg.s
    ## Data
    T = 16
    t = 16
    ## Model
    model = MINC.ConvBlock(T, t, chs)
    ## Setup
    rng = cfg.PRNG(cfg.seed)
    ps, st = Lux.setup(Lux.replicate(rng), model)
    ## Transfer
    ps = ComponentArray(ps) |> dev
    st = st |> dev
    ## Input
    x = randn(Float32, T, cfg.r * cfg.s, cfg.batch_size * bs) |> dev
    return x, model, ps, st, cfg
end
