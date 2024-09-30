"""
Appends to either StackA or StackZ the appropriate head that outputs a image output with
the appropriate transformation properties under symmetry transformations
"""
## IMAGER
struct Imager{C <: Lux.AbstractExplicitContainerLayer, I <: Int} <:
       Lux.AbstractExplicitContainerLayer{(:chain,)}
    chain::C
    grid_length::I
end
##
function ImagerA(; cfg=Config())
    ## Device
    dev = cfg.dev
    ordG = cfg.ordG
    # Time index length
    t_start = cfg.t_start
    t_end = cfg.t_end
    t_start_compressed, t_end_compressed = time_compress(t_start, t_end, cfg)
    T = length(t_start_compressed:t_end_compressed)
    ## Stack
    _stack = StackA(T; cfg=cfg)
    # Connect
    T_out = _stack.T_out
    chs = cfg.A_chs
    L = cfg.A_L
    grid_length = cfg.grid_length
    connect = ToGrid{Val(L)}(T_out, ordG, chs, grid_length)
    # CHAIN
    chain = Chain(_stack, connect)
    return Imager(chain, cfg.grid_length)
end
##
function ImagerZ(; cfg=Config())
    ## Device
    dev = cfg.dev
    # Time index length
    t_start = cfg.t_start
    t_end = cfg.t_end
    t_start_compressed, t_end_compressed = time_compress(t_start, t_end, cfg)
    T = length(t_start_compressed:t_end_compressed)
    ## Stack
    _stack = StackZ(T; cfg=cfg)
    # Connect
    T_out = _stack.T_out
    chs = cfg.Z_chs
    activation = cfg.Z_connect_activation
    grid_length = cfg.grid_length
    connect = ToNumberField(T_out, chs, grid_length)
    chain = Chain(_stack, connect)
    return Imager(chain, cfg.grid_length)
end
## Train
function (m::Imager)(
        x::AbstractArray{Float32, 4}, ps::ComponentArray, st::NamedTuple)
    # Train: size(x) = (time, receiver, sender, batch)
    x_chain, st_chain = m.chain(x, ps, st)
    return x_chain, st_chain
end
## Test
function (m::Imager)(
        x::AbstractArray{Float32, 5}, ps::ComponentArray, st::NamedTuple)
    # Test: size(x) = (time, receiver, sender, baseline, batch)
    x_r = reshape(x, size(x)[1:3]..., :) #[t, r, s, bl * b]
    ŷ_p_r, st_chain = m.chain(x_r, ps, st) #[x, y, bl * b]
    ŷ_p = reshape(ŷ_p_r, size(ŷ_p_r)[1:2]..., size(x)[4:5]...) #[x, y, bl, b]
    #
    _ŷ_p = permutedims(ŷ_p, (1, 2, 4, 3)) #[i, b, bl]
    m_x = dropdims(mean(_ŷ_p; dims=4); dims=4) #[i, b]
    return m_x, st_chain
end
##
function get_imager(cfg::Config)
    model_type = cfg.model_type
    if model_type == "A"
        model = ImagerA(; cfg=cfg)
    elseif cfg.model_type == "Z"
        model = ImagerZ(; cfg=cfg)
    end
    return model
end
