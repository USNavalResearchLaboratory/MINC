## UTIL FOR TESTS
struct ImagerATest{F <: Function, S <: String} <: TimeDefToGrid
    getModel::F
    name::S
end
##
function ImagerATest()
    return ImagerATest(getImagerA, "ImagerA")
end
##
function getImagerA(; bs=1, kwargs...)
    ## Config
    cfg = MINC.Config(; epochs=2, infotime=1, batch_size=2, kwargs...)
    ## Device
    dev = cfg.dev
    # Time index length
    t_start = cfg.t_start
    t_end = cfg.t_end
    t_start_compressed, t_end_compressed = MINC.time_compress(
        t_start, t_end, cfg)
    T = length(t_start_compressed:t_end_compressed)
    ## Data
    r = cfg.r
    s = cfg.s
    batch_size = cfg.batch_size
    ## Model
    cfg.grid_length = 5
    model = MINC.ImagerA(; cfg=cfg)
    ## Setup
    rng = cfg.PRNG(cfg.seed)
    ps, st = Lux.setup(Lux.replicate(rng), model)
    ## Transfer
    ps = ComponentArray(ps) |> dev
    st = st |> dev
    ## Input
    x = randn(Float32, T, r, s, batch_size * bs) |> dev
    ##
    return x, model, ps, st, cfg
end
