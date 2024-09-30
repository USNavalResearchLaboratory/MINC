##
function delete_zero_mode(voltage_series)
    # Voltage shifts by a constant are nonphysical; set origin to zero
    return voltage_series .- mean(voltage_series)
end
##
function zero_diagonals!(signals::Array{Float32, N}) where {N}
    for i in 1:4
        signals[:, i, i, fill(:, N - 3)...] .= 0
    end
    return signals
end
##
function window_signals(
        input_data::Array{Float32, N}, t_start, t_end; cfg=Config()) where {N}
    t_start_compressed, t_end_compressed = time_compress(t_start, t_end, cfg)
    input_data = input_data[
        t_start_compressed:t_end_compressed, fill(:, N - 1)...]
    return input_data
end
