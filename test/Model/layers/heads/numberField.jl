## UTIL FOR TESTS
struct ToNumberFieldTest{F <: Function, S <: String} <: RToR
    getModel::F
    name::S
end
##
function ToNumberFieldTest()
    return ToNumberFieldTest(getToNumberField, "ToNumberField")
end
##
function getToNumberField(; bs=1, kwargs...)
    ## Config
    cfg = MINC.Config(; epochs=2, infotime=1, batch_size=2, kwargs...)
    ## Device
    dev = cfg.dev
    ## Channels
    chs = 16
    grid_length = 5
    ## Data
    T = 4
    ordG = cfg.ordG
    batch_size = cfg.batch_size
    ## Model
    model = MINC.ToNumberField(T, chs, grid_length)
    ## Setup
    rng = cfg.PRNG(cfg.seed)
    ps, st = Lux.setup(Lux.replicate(rng), model)
    ## Transfer
    ps = ComponentArray(ps) |> dev
    st = st |> dev
    ## Input
    x = randn(Float32, T, chs, batch_size * bs) |> dev
    ##
    return x, model, ps, st, cfg
end
