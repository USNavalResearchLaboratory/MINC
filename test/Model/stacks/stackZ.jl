## UTIL FOR TESTS
struct StackZTest{F <: Function, S <: String} <: RToR
    getModel::F
    name::S
end
##
function StackZTest()
    return StackZTest(getStackZ, "StackZ")
end
##
function getStackZ(; bs=1, kwargs...)
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
    ##
    batch_size = cfg.batch_size
    r = cfg.r
    s = cfg.s
    ## Model
    model = MINC.StackZ(T; cfg=cfg)
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
