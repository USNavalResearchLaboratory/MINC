"""
Transforms a model output of RegBlock in the regular representation to features that transform under p4m
like images. Parameterized by P which dictates whether the type is exactly equivariant or approximately equivariant.
"""
## ToGrid
Lux.@concrete struct ToGrid{P} <: Lux.AbstractExplicitLayer
    T::Int
    ordG::Int
    chs::Int
    grid_length::Int
end
##
function Lux.initialparameters(rng::AbstractRNG, m::ToGrid{Val(0)})
    ##
    weight = randn(rng::AbstractRNG, Float32,
        m.grid_length * m.grid_length, m.T * m.chs) .*
             Float32(sqrt(2 / (m.T * m.chs * m.ordG)))
    ##
    return (; weight)
end
function Lux.initialparameters(rng::AbstractRNG, m::ToGrid{Val(1)})
    ##
    weight = randn(rng::AbstractRNG, Float32,
        m.grid_length * m.grid_length, m.T * m.chs) .*
             Float32(sqrt(2 / (m.T * m.chs * m.ordG)))
    weight_g = zeros(Float32, m.ordG)
    ##
    return (; weight, weight_g)
end
##
function Lux.initialstates(rng::AbstractRNG, m::ToGrid)
    # Group indices
    # [i, x, y, g]
    grid_rep = get_grid_rep(m.grid_length)
    perms = reshape(grid_rep, 2, :, m.ordG)
    return (; perms)
end
##
function Lux.parameterlength(m::ToGrid{Val(0)})
    return m.grid_length * m.grid_length * m.T * m.chs
end
function Lux.parameterlength(m::ToGrid{Val(1)})
    return m.grid_length * m.grid_length * m.T * m.chs + m.ordG
end
##
function Base.show(io::IO, m::ToGrid)
    print(io, "ToGrid(grid_length = ", m.grid_length)
    return print(io, ")")
end
##
function (m::ToGrid{Val(0)})(
        x::AbstractArray{Float32, 4}, ps::ComponentArray, st::NamedTuple)
    ## [t, c, g, b]
    x_p = permutedims(x, (1, 3, 2, 4))
    ## [t * c, g, b]
    x_p_r = reshape(x_p, :, size(x, 2), size(x, 4))
    ## [x * y, g, b]
    x_dense_r = batched_mul(ps.weight, x_p_r)
    x_dense = reshape(
        x_dense_r, m.grid_length, m.grid_length, size(x, 2), size(x, 4))
    ## [g][i, x * y]
    perms_g = _eachslice(st.perms, Val(3))
    #####
    x_grid_r = sum(map(
        g -> stack(
            map(ind -> x_dense[CartesianIndex(Tuple(ind)), g, :],
                _eachslice(perms_g[g], Val(2)));
            dims=1),
        1:(m.ordG)))
    ##
    x_softmax = softmax(x_grid_r)
    m_x = reshape(x_softmax, m.grid_length, m.grid_length, size(x, 4))
    ##
    return m_x, st
end
##
function (m::ToGrid{Val(1)})(
        x::AbstractArray{Float32, 4}, ps::ComponentArray, st::NamedTuple)
    ## [t, c, g, b]
    x_p = permutedims(x, (1, 3, 2, 4))
    ## [t * c, g, b]
    x_p_r = reshape(x_p, :, size(x, 2), size(x, 4))
    ## [x * y, g, b]
    x_dense_r = batched_mul(ps.weight, x_p_r)
    _x_dense = reshape(
        x_dense_r, m.grid_length, m.grid_length, size(x, 2), size(x, 4))
    ##
    w_g = m.ordG .* softmax(ps.weight_g ./ m.ordG)
    w_g_r = reshape(w_g, 1, 1, m.ordG, 1)
    ##
    x_dense = w_g_r .* _x_dense
    ## [g][i, x * y]
    perms_g = _eachslice(st.perms, Val(3))
    x_grid_r = sum(map(
        g -> stack(
            map(ind -> x_dense[CartesianIndex(Tuple(ind)), g, :],
                _eachslice(perms_g[g], Val(2)));
            dims=1),
        1:(m.ordG)))
    x_softmax = softmax(x_grid_r)
    m_x = reshape(x_softmax, m.grid_length, m.grid_length, size(x, 4))
    ##
    return m_x, st
end
##
