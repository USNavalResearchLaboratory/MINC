"""
Symmetry-breaking weight layer.
"""
## Approximate Equivariant Biasing Layer
Lux.@concrete struct ApproxLayer{P} <: Lux.AbstractExplicitLayer
    ordG::Int
end
##
function Lux.initialparameters(rng::AbstractRNG, m::ApproxLayer)
    weight = zeros(Float32, m.ordG)
    return (; weight)
end
## size(x) = (t, g, c, b)
function (m::ApproxLayer{Val(0)})(
        x::AbstractArray{Float32, 4}, ps, st::NamedTuple)
    #
    _w = ps.weight
    ordG = size(_w, 1)
    w = ordG .* softmax(_w ./ ordG)
    #
    x_p = permutedims(x, (2, 1, 3, 4)) #[g, t, c, b]
    m_x_p = w .* x_p
    m_x = permutedims(m_x_p, (2, 1, 3, 4))
    return m_x, st
end
## size(x) = (c, t, g, b)
function (m::ApproxLayer{Val(1)})(
        x::AbstractArray{Float32, 4}, ps, st::NamedTuple)
    #
    _w = ps.weight
    ordG = size(_w, 1)
    w = ordG .* softmax(_w ./ ordG)
    #
    x_p = permutedims(x, (3, 2, 1, 4)) #[g, t, c, b] 
    m_x_p = w .* x_p
    m_x = permutedims(m_x_p, (3, 2, 1, 4))
    return m_x, st
end
## size(x) = (t * c, g, b)
function (m::ApproxLayer{Val(2)})(
        x::AbstractArray{Float32, 3}, ps, st::NamedTuple)
    #
    _w = ps.weight
    ordG = size(_w, 1)
    w = ordG .* softmax(_w ./ ordG)
    #
    x_p = permutedims(x, (2, 1, 3)) #[g, t * c, b]
    m_x_p = w .* x_p
    m_x = permutedims(m_x_p, (2, 1, 3)) #[t * c, g, b]
    return m_x, st
end
