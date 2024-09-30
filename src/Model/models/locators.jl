"""
Appends to either StackA or StackZ the appropriate head that outputs a two-component output with
the appropriate transformation properties under symmetry transformations
"""
## LOCATORS
struct Locator{C <: Lux.AbstractExplicitContainerLayer, F <: Function} <:
       Lux.AbstractExplicitContainerLayer{(:chain,)}
    chain::C
    loss::F
end
##
function LocatorA(; cfg=Config())
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
    activation = cfg.A_connect_activation
    connect = ToVector(
        T_out, ordG, chs, ordG * chs; activation=activation, L=L, dev=dev)
    # CHAIN
    chain = Chain(_stack, connect)
    return Locator(chain, cfg.loss)
end
##
function LocatorZ(; cfg=Config())
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
    #
    connect = ToTwoNumbers(T_out, chs, chs; activation=activation)
    chain = Chain(_stack, connect)
    return Locator(chain, cfg.loss)
end
## Train
function (m::Locator)(
        x::AbstractArray{Float32, 4}, ps::ComponentArray, st::NamedTuple)
    # Train: size(x) = (time, receiver, sender, batch)
    x_chain, st_chain = m.chain(x, ps, st)
    return x_chain, st_chain
end
## Test
function (m::Locator)(
        x::AbstractArray{Float32, 5}, ps::ComponentArray, st::NamedTuple)
    # Test: size(x) = (time, receiver, sender, baseline, batch)
    x_r = reshape(x, size(x)[1:3]..., :) #[t, r, s, bl * b]
    ŷ_p_r, st_chain = m.chain(x_r, ps, st) #[i, bl * b]
    ŷ_p = reshape(ŷ_p_r, size(ŷ_p_r, 1), size(x)[4:5]...) #[i, bl, b]
    #
    _ŷ_p = permutedims(ŷ_p, (1, 3, 2)) #[i, b, bl]
    m_x = dropdims(mean(_ŷ_p; dims=3); dims=3) #[i, b]
    return m_x, st_chain
end
##
function get_locator(cfg::Config)
    model_type = cfg.model_type
    if model_type == "A"
        model = LocatorA(; cfg=cfg)
    elseif cfg.model_type == "Z"
        model = LocatorZ(; cfg=cfg)
    end
    return model
end
