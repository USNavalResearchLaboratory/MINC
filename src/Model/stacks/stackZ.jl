"""
Chains together the Convolutional Blocks used by the ordinary model
"""
## STACK Z
struct StackZ{C <: Lux.AbstractExplicitContainerLayer, I <: Integer} <:
       Lux.AbstractExplicitContainerLayer{(:chain,)}
    chain::C
    T_out::I
end
##
function StackZ(T; cfg=Config())
    ## Unpack
    #
    activation = cfg.Z_activation
    p = cfg.Z_p
    chs = cfg.Z_chs
    #
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
    layer_1 = ConvBlock(T_1, T_1, 16 => chs; pad=pad_1)
    layer_2 = ConvBlock(T_2, T_2, chs; pad=pad_2, activation=activation, p=p)
    layer_3 = ConvBlock(T_3, T_3, chs; pad=pad_3, activation=activation, p=p)
    layer_4 = ConvBlock(T_4, T_4, chs; pad=pad_4, activation=activation, p=p)
    layer_5 = ConvBlock(T_5, T_5, chs; pad=pad_5, activation=activation, p=p)
    chain = Chain(layer_1, layer_2, layer_3, layer_4, layer_5)
    #
    T_out = T_6
    return StackZ(chain, T_6)
end
##
function (m::StackZ)(
        x::AbstractArray{Float32, 4}, ps::ComponentArray, st::NamedTuple)
    x_chain, st_chain = m.chain(x, ps, st)
    return x_chain, st_chain
end
