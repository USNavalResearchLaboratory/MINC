"""
Transforms an output of RegBlock to a two-component scalar output suitable for detection.
If L = 0, then the layer is exactly equivariant; otherwise, it is approximately equivariant.
"""
## BINARY
Lux.@concrete struct ToTwoScalars <:
                     Lux.AbstractExplicitContainerLayer{(:dense_in, :dense_out)}
    dense_in::Lux.AbstractExplicitLayer
    dense_out::Lux.AbstractExplicitLayer
end
##
function ToTwoScalars(T::Int, ordG::Int, chs_1::Int, chs_2::Int;
        activation=identity, L=0, init_weight=Lux.kaiming_normal)
    ##
    dense = Dense(T * chs_1 => 2 * chs_2, activation;
        init_weight=init_weight, use_bias=false)
    if L == 0
        dense_in = dense
    else
        approx_layer = ApproxLayer{Val(2)}(ordG)
        dense_in = Chain(dense, approx_layer)
    end
    dense_out = Dense(chs_2 => 1; init_weight=init_weight, use_bias=false)
    return ToTwoScalars(dense_in, dense_out)
end
##
function (m::ToTwoScalars)(
        x::AbstractArray{Float32, 4}, ps::ComponentArray, st::NamedTuple)
    ##
    x_p_r = permutedims(x, (1, 3, 2, 4)) #[t, c, g, b]
    x_p = reshape(x_p_r, :, size(x, 2), size(x, 4)) #[t * c, g, b]
    ##
    x_dense_in, st_dense_in = m.dense_in(x_p, ps.dense_in, st.dense_in) #[c * 2, g, b]
    kaiming_g_norm = Float32(sqrt(2 / size(x, 2)))
    x_g_invariant = dropdims(sum(x_dense_in; dims=2); dims=2) .* kaiming_g_norm #[c * 2, b]
    x_r = reshape(x_g_invariant, :, 2, size(x, 4)) #[c, 2, b]
    ##
    _m_x, st_dense_out = m.dense_out(x_r, ps.dense_out, st.dense_out) #[1, 2, b]
    m_x = dropdims(_m_x; dims=1) #[2, b]
    ##
    st_ = (dense_in=st_dense_in, dense_out=st_dense_out)
    return m_x, st_
end
