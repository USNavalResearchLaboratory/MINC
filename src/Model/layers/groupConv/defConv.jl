"""
Defining (lifting) group convolution.
"""
## Defining Conv
Lux.@concrete struct DefConv <: Lux.AbstractExplicitLayer
    chs_in::Int
    chs_out::Int
    kernel_size::Tuple{Int}
    pad::NTuple{2, Int}
    groups::Int
    init_weight::Function
    ordG::Int
    r::Int
    s::Int
end
##
function DefConv(k::Tuple{Int}, ordG::Int, ch::Pair{Int, Int};
        init_weight=kaiming_normal, pad::NTuple{2, Int}=(0, 0),
        groups::Int=1, r::Int=4, s::Int=4)
    return DefConv(first(ch), last(ch), k, pad, groups, init_weight, ordG, r, s)
end
##
function Lux.initialparameters(rng::AbstractRNG, m::DefConv)
    weight = _convfilter(
        rng::AbstractRNG, m.kernel_size, m.r * m.s * m.chs_in => m.chs_out;
        init=m.init_weight, groups=m.groups)
    return (; weight)
end
function Lux.initialstates(rng::AbstractRNG, m::DefConv)
    perms = get_defining_perms(m.ordG) # 4x4 Matrix{Int64}
    return (; perms)
end
##
function (m::DefConv)(x::AbstractArray, ps, st::NamedTuple)
    G_w = defG_w(st.perms, ps.weight, m.r, m.s) #[t, r * s * div(c_in, groups), ordG * c_out]
    x_r = reshape(x, size(x, 1), :, size(x, ndims(x)))
    cdims = DenseConvDims(
        x_r, G_w; stride=(1,), padding=m.pad, dilation=(1,), groups=m.groups)
    x_conv = NNlib.conv(x_r, G_w, cdims)
    m_x = reshape(x_conv, size(x_conv, 1), m.ordG, m.chs_out, size(x, ndims(x)))
    return m_x, st
end
##
function Base.show(io::IO, m::DefConv)
    print(io, "DefConv((", m.kernel_size[1], ", ", m.ordG, ", ", m.ordG, ")")
    print(io, ", ", m.chs_in, " => ", m.chs_out)
    _print_conv_opt(io, m)
    return print(io, ")")
end
function _print_conv_opt(io::IO, m::DefConv)
    all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
    (m.groups == 1) || print(io, ", groups=", m.groups)
    return nothing
end
## UTILITIES
@inline function defG_w(perms::AbstractArray{Int, 2},
        w::AbstractArray{Float32, 3}, r::Int, s::Int)
    ##[t, r, s, div(chs_in, groups), chs_out]
    w_r = reshape(w, size(w, 1), r, s, :, size(w, 3))
    ## Rotates w through the group
    ##[ordG][t, r * s * div(chs_in, groups), chs_out]
    _G_w_r = @views map(
        perm -> w_r[:, perm, perm, :, :], _eachslice(perms, Val(2)))
    G_w_r = stack(_G_w_r; dims=5) #[t, r, s, div(chs_in, groups), ordG, chs_out]
    ##[t, r * s * div(chs_in, groups), ordG * chs_out]
    G_w = reshape(G_w_r, size(G_w_r, 1), prod(size(G_w_r)[2:4]), :)
    ##
    return G_w
end
