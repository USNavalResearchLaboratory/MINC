## UTIL FOR TESTS
struct ToGridTest{F <: Function, S <: String} <: TimeRegToGrid
    getModel::F
    name::S
end
##
function ToGridTest()
    return ToGridTest(getToGrid, "ToGrid")
end
##
function getToGrid(; bs=1, kwargs...)
    ## Config
    cfg = MINC.Config(; epochs=2, infotime=1, batch_size=2, kwargs...)
    ## Device
    dev = cfg.dev
    ## Channels
    chs = 16
    cfg.grid_length = 5
    ## Data
    T = 4
    ordG = cfg.ordG
    batch_size = cfg.batch_size
    L = cfg.A_L
    ## Model
    model = MINC.ToGrid{Val(L)}(T, ordG, chs, cfg.grid_length)
    ## Setup
    rng = cfg.PRNG(cfg.seed)
    ps, st = Lux.setup(Lux.replicate(rng), model)
    ## Transfer
    ps = ComponentArray(ps) |> dev
    st = st |> dev
    ## Input
    x = randn(Float32, T, ordG, chs, batch_size * bs) |> dev
    ##
    return x, model, ps, st, cfg
end
