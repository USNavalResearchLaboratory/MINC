"""
Adds activation and LayerNorm to the ordinary Conv layer
"""
## CONV BLOCK
struct ConvBlock{C <: Lux.AbstractExplicitLayer} <:
       Lux.AbstractExplicitContainerLayer{(:chain,)}
    chain::C
end
##
function ConvBlock(T::Int, t::Int, chs::Int; pad::NTuple{2, Int}=(0, 0),
        groups::Int=1, init_weight::Function=kaiming_normal,
        p::Float32=0.0f0, activation::Function=identity)
    # Time Sequence Length
    T_out = T - (t - 1) + sum(pad)
    # Conv
    conv = Conv((t,), chs => chs; groups=groups, pad=pad,
        init_weight=init_weight, use_bias=false)
    dropout = Dropout(p; dims=2)
    # Norm
    norm = LayerNorm((T_out, chs); affine=false, dims=(2,))
    # Skip
    pooler = Pooler{Val(0)}((t,), chs; pad=pad)
    conv_skip = Parallel(+, conv, pooler)
    # Chain
    chain = Chain(conv_skip, norm, activation, dropout)
    return ConvBlock(chain)
end
function ConvBlock(
        T::Int, t::Int, ch::Pair{Int, Int}; pad::NTuple{2, Int}=(0, 0),
        groups::Int=1, init_weight::Function=kaiming_normal)
    # Time Sequence Length
    T_out = T - (t - 1) + sum(pad)
    # Channels
    chs_in = ch[1]
    chs_out = ch[2]
    # Conv
    conv = Conv((t,), chs_in => chs_out; groups=groups, pad=pad,
        init_weight=init_weight, use_bias=false)
    chain = Chain(conv)
    ##
    return ConvBlock(chain)
end
##
function (m::ConvBlock)(
        x::AbstractArray{Float32, 3}, ps::ComponentArray, st::NamedTuple)
    return m.chain(x, ps, st)
end
function (m::ConvBlock)(
        x::AbstractArray{Float32, 4}, ps::ComponentArray, st::NamedTuple)
    x_r = reshape(x, size(x, 1), :, size(x, 4))
    return m.chain(x_r, ps, st)
end
