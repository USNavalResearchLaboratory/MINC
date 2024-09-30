"""
Transforms the ordinary channel index of a ConvBlock into a grid suitable for imaging.
"""
## ToNumberField
Lux.@concrete struct ToNumberField <: Lux.AbstractExplicitLayer
    T::Int
    chs::Int
    grid_length::Int
end
##
function Lux.initialparameters(rng::AbstractRNG, m::ToNumberField)
    ##
    weight = randn(rng::AbstractRNG, Float32,
        m.grid_length * m.grid_length, m.T * m.chs) .*
             Float32(sqrt(2 / (m.T * m.chs)))
    ##
    return (; weight)
end
##
function Lux.initialstates(rng::AbstractRNG, m::ToNumberField)
    return (;)
end
##
function Lux.parameterlength(m::ToNumberField)
    return m.grid_length * m.grid_length * m.T * m.chs
end
##
function Base.show(io::IO, m::ToNumberField)
    print(io, "ToNumberField(grid_length = ", m.grid_length)
    return print(io, ")")
end
##
function (m::ToNumberField)(
        x::AbstractArray{Float32, 3}, ps::ComponentArray, st::NamedTuple)
    ## [t * c, b]
    x_r = reshape(x, :, size(x, 3))
    ## [x * y, b]
    x_dense_r = batched_mul(ps.weight, x_r)
    ##
    x_softmax = softmax(x_dense_r)
    m_x = reshape(x_softmax, m.grid_length, m.grid_length, size(x, 3))
    ##
    return m_x, st
end
##
