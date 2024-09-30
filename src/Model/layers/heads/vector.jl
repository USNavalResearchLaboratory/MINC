"""
Transforms an output of RegBlock to a two-component vector output suitable for localization.
If L = 0, then the layer is exactly equivariant; otherwise, it is approximately equivariant.
"""
## VECTOR
Lux.@concrete struct ToVector <:
                     Lux.AbstractExplicitContainerLayer{(:dense_in, :dense_out)}
    dense_in::Lux.AbstractExplicitLayer
    dense_out::Lux.AbstractExplicitLayer
    rot_NamedTuple::NamedTuple
end
##
function ToVector(
        T::Int, ordG::Int, chs_1::Int, chs_2::Int; activation=identity,
        L=0, init_weight::Function=Lux.kaiming_uniform, dev=gpu_device())
    # Prepare rotation matrix
    rot_Array = get_vec_rep(ordG) #[g, i, j]
    rot_Array_p = permutedims(rot_Array, (2, 3, 1)) #[j, i, g]
    rot_Array_p_r = reshape(rot_Array_p, size(rot_Array_p, 1), :) #[j, i * g]
    rot_NamedTuple = (rot_Array_p_r=rot_Array_p_r,) |> dev
    _dense_in = Dense(T * chs_1 => 2 * chs_2, activation;
        init_weight=init_weight, use_bias=false)
    if L == 0
        dense_in = _dense_in
    else
        approx_layer = ApproxLayer{Val(2)}(ordG)
        dense_in = Chain(_dense_in, approx_layer)
    end
    dense_out = Dense(chs_2 => 1; init_weight=init_weight, use_bias=false)
    return ToVector(dense_in, dense_out, rot_NamedTuple)
end
##
function (m::ToVector)(
        x::AbstractArray{Float32, 4}, ps::ComponentArray, st::NamedTuple)
    ##
    x_p_r = permutedims(x, (1, 3, 2, 4)) #[t, c, g, b]
    x_p = reshape(x_p_r, :, size(x, 2), size(x, 4)) #[t * c, g, b]
    ##
    x_dense_in, st_dense_in = m.dense_in(x_p, ps.dense_in, st.dense_in) #[c * 2, g, b]
    x_dense_in_r = reshape(x_dense_in, :, 2 * size(x, 2), size(x, 4)) #[c, 2 * g, b]
    x_dense_in_r_p = permutedims(x_dense_in_r, (2, 1, 3)) #[2 * g, c, b]
    ##
    R_p_r = m.rot_NamedTuple.rot_Array_p_r #[j, i * g]
    kaiming_g_norm = Float32(sqrt(2 / size(x, 2)))
    R_x_p = batched_mul(R_p_r, x_dense_in_r_p) .* kaiming_g_norm #[j, c, b]
    R_x = permutedims(R_x_p, (2, 1, 3)) #[c, j, b]
    _m_x, st_dense_out = m.dense_out(R_x, ps.dense_out, st.dense_out) #[1, j, b]
    m_x = dropdims(_m_x; dims=1) #[j, b]
    ##
    st_ = (dense_in=st_dense_in, dense_out=st_dense_out)
    ##
    return m_x, st_
end
