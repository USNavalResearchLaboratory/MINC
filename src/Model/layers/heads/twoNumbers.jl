"""
Transforms an output of ConvBlock to a binary output suitable for detection.
"""
## NUMBER
Lux.@concrete struct ToTwoNumbers <:
                     Lux.AbstractExplicitContainerLayer{(:dense_in, :dense_out)}
    dense_in::Lux.AbstractExplicitLayer
    dense_out::Lux.AbstractExplicitLayer
end
##
function ToTwoNumbers(T::Int, chs_1::Int, chs_2::Int; activation=identity,
        init_weight::Function=Lux.kaiming_normal)
    dense_in = Dense(
        T * chs_1 => 2 * chs_2, activation; init_weight=init_weight)
    dense_out = Dense(chs_2 => 1; init_weight=init_weight)
    return ToTwoNumbers(dense_in, dense_out)
end
##
function (m::ToTwoNumbers)(
        x::AbstractArray{Float32, 3}, ps::ComponentArray, st::NamedTuple)
    ##
    x_r = reshape(x, :, size(x, 3))
    ##
    x_dense_in, st_dense_in = m.dense_in(x_r, ps.dense_in, st.dense_in) #[c * 2, b]
    x_dense_in_r = reshape(x_dense_in, :, 2, size(x, 3)) #[c, 2, b]
    _m_x, st_dense_out = m.dense_out(x_dense_in_r, ps.dense_out, st.dense_out) #[1, 2, b]
    m_x = dropdims(_m_x; dims=1) #[2, b]
    ##
    st_ = (dense_in=st_dense_in, dense_out=st_dense_out)
    return m_x, st_
end
##
