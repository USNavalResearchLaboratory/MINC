"""
Non-trainable averaging kernel on the time index
"""
## Pooling
Lux.@concrete struct Pooler{P} <: Lux.AbstractExplicitLayer
    chs::Int
    kernel_size::Tuple{Int}
    pad::NTuple{2, Int}
end
##
function Pooler{P}(
        k::Tuple{Int}, chs::Int; pad::NTuple{2, Int}=(0, 0)) where {P}
    return Pooler{P}(chs, k, pad)
end
##
function Lux.initialparameters(rng::AbstractRNG, m::Pooler)
    return (;)
end
function Lux.initialstates(rng::AbstractRNG, m::Pooler)
    ##
    weight = _convfilter(rng::AbstractRNG, m.kernel_size, m.chs => m.chs;
        init=kaiming_normal, groups=m.chs)
    weight .= 1.0f0
    weight = weight ./ size(weight, 1)
    ##
    return (; weight)
end
##
function (m::Pooler{Val(0)})(x::AbstractArray{Float32, 3}, ps, st::NamedTuple)
    ##
    cdims = DenseConvDims(
        x, st.weight; stride=(1,), padding=m.pad, dilation=(1,), groups=m.chs)
    x_pool = NNlib.conv(x, st.weight, cdims) #[t, c, b]
    ##
    return x_pool, st
end
function (m::Pooler{Val(1)})(x::AbstractArray{Float32, 4}, ps, st::NamedTuple)
    ##
    x_r = reshape(x, size(x, 1), :, size(x, 4)) #[t, ordG * c, b]
    ##
    cdims = DenseConvDims(
        x_r, st.weight; stride=(1,), padding=m.pad, dilation=(1,), groups=m.chs)
    ##
    x_pool_r = NNlib.conv(x_r, st.weight, cdims) #[t, ordG * c, b]
    ##
    x_pool = reshape(x_pool_r, size(x_pool_r, 1), size(x)[2:4]...)
    ##
    return x_pool, st
end
