"""
Appends to either StackA or StackZ the appropriate head that outputs a binary output with
the appropriate transformation properties under symmetry transformations
"""
## DETECTOR
struct Detector{C <: Lux.AbstractExplicitContainerLayer} <:
       Lux.AbstractExplicitContainerLayer{(:chain,)}
    chain::C
end
##
function DetectorA(; cfg=Config())
    #
    ordG = cfg.ordG
    # Time index length
    t_start = cfg.t_start
    t_end = cfg.t_end
    t_start_compressed, t_end_compressed = time_compress(t_start, t_end, cfg)
    T = length(t_start_compressed:t_end_compressed)
    # Stack
    _stack = StackA(T; cfg=cfg)
    # Connect
    T_out = _stack.T_out
    chs = cfg.A_chs
    L = cfg.A_L
    activation = cfg.A_activation
    connect = ToTwoScalars(
        T_out, ordG, chs, ordG * chs; activation=activation, L=L)
    # CHAIN
    chain = Chain(_stack, connect)
    return Detector(chain)
end
function DetectorZ(; cfg=Config())
    # Time index length
    t_start = cfg.t_start
    t_end = cfg.t_end
    t_start_compressed, t_end_compressed = time_compress(t_start, t_end, cfg)
    T = length(t_start_compressed:t_end_compressed)
    # Stack
    _stack = StackZ(T; cfg=cfg)
    # Connect
    T_out = _stack.T_out
    chs = cfg.Z_chs
    activation = cfg.Z_activation
    # CHAIN
    connect = ToTwoNumbers(T_out, chs, chs; activation=activation)
    chain = Chain(_stack, connect)
    return Detector(chain)
end
## Train
function (m::Detector)(
        x::AbstractArray{Float32, 4}, ps::ComponentArray, st::NamedTuple)
    # Train: size(x) = (time, receiver, sender, batch)
    x_chain, st_chain = m.chain(x, ps, st)
    return x_chain, st_chain
end
##
function get_detector(cfg)
    # MODEL
    model_type = cfg.model_type
    if model_type == "A"
        model = DetectorA(; cfg=cfg)
    elseif cfg.model_type == "Z"
        model = DetectorZ(; cfg=cfg)
    end
    return model
end
