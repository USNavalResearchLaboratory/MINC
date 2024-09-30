"""
Chains together Defining and Regular Blocks used by the Equivariant and Approximately
Equivariant Models.
"""
## STACK A
struct StackA{C <: Lux.AbstractExplicitContainerLayer, I <: Integer} <:
       Lux.AbstractExplicitContainerLayer{(:chain,)}
    chain::C
    T_out::I
end
##
function StackA(T; cfg=Config())
    ## Unpack
    ordG = cfg.ordG
    activation = cfg.A_activation
    p = cfg.A_p
    chs = cfg.A_chs
    L = cfg.A_L
    ## Reducing sequence length
    T_1 = div(T, 2^0, RoundUp)
    T_2 = div(T, 2^1, RoundUp)
    T_3 = div(T, 2^2, RoundUp)
    T_4 = div(T, 2^3, RoundUp)
    T_5 = div(T, 2^4, RoundUp)
    T_6 = div(T, 2^5, RoundUp)
    #
    pad_1 = get_half_pad(T_1)
    @assert T_2 == sum(pad_1) + 1
    #
    pad_2 = get_half_pad(T_2)
    @assert T_3 == sum(pad_2) + 1
    #
    pad_3 = get_half_pad(T_3)
    @assert T_4 == sum(pad_3) + 1
    #
    pad_4 = get_half_pad(T_4)
    @assert T_5 == sum(pad_4) + 1
    #
    pad_5 = get_half_pad(T_5)
    @assert T_6 == sum(pad_5) + 1
    ##
    # CHAIN
    layer_1 = DefBlock(T_1, T_1, ordG, 1 => chs; pad=pad_1, L=L)
    layer_2 = RegBlock(
        T_2, T_2, ordG, chs; pad=pad_2, L=L, activation=activation, p=p)
    layer_3 = RegBlock(
        T_3, T_3, ordG, chs; pad=pad_3, L=L, activation=activation, p=p)
    layer_4 = RegBlock(
        T_4, T_4, ordG, chs; pad=pad_4, L=L, activation=activation, p=p)
    layer_5 = RegBlock(
        T_5, T_5, ordG, chs; pad=pad_5, L=L, activation=activation, p=p)
    chain = Chain(layer_1, layer_2, layer_3, layer_4, layer_5)
    #
    T_out = T_6
    return StackA(chain, T_6)
end
##
function (m::StackA)(
        x::AbstractArray{Float32, 4}, ps::ComponentArray, st::NamedTuple)
    x_chain, st_chain = m.chain(x, ps, st)
    return x_chain, st_chain
end
