"""
Adds ApproxLayer to the lifting RegConv if L != 0. Also introduces pooling, skip, LayerNorm, and activation.
"""

## MULTIHEAD BLOCK
struct RegBlock{C} <: Lux.AbstractExplicitContainerLayer{(:chain,)}
    chain::C
end
##
function RegBlock(T::Int, t::Int, ordG::Int, chs::Int; L::Int=0,
        pad::NTuple{2, Int}=(0, 0), groups::Int=1,
        init_weight::Function=kaiming_normal,
        p::Float32=0.0f0, activation::Function=identity)
    # Time Sequence Length
    T_out = T - (t - 1) + sum(pad)
    # Conv
    reg_conv = RegConv(
        (t,), ordG, chs => chs; init_weight=init_weight, pad=pad, groups=groups)
    if L == 0
        conv = reg_conv
    else
        approx_layer = ApproxLayer{Val(0)}(ordG)
        conv = Chain(reg_conv, approx_layer)
    end
    # Pooling
    dropout = Dropout(p; dims=3)
    # Norm
    norm = LayerNorm((T_out, ordG, chs); affine=false, dims=(2, 3))
    # Skip
    pooler = Pooler{Val(1)}((t,), ordG * chs; pad=pad)
    conv_skip = Parallel(+, conv, pooler)
    # Chain
    chain = Chain(conv_skip, norm, activation, dropout)
    return RegBlock(chain)
end
##
function (m::RegBlock)(
        x::AbstractArray{Float32, 4}, ps::ComponentArray, st::NamedTuple)
    m_x, st_ = m.chain(x, ps, st)
    return m_x, st_
end
