## PROCESS DATA AFTER BASELINE SUBTRACTION
function post_process_locate_data(input_data, target_data; σ_maxabs::Real=0.43)
    ## Zero Mean
    input_data_zm = stack(map(signal -> signal .- mean(signal),
        eachslice(input_data; dims=(2, 3, 4, 5))))
    ## Discard max absolute outliers
    inputs_good, targets_good = maxabs_acceptable_signals_blsub(
        input_data_zm, target_data; σ_maxabs=σ_maxabs)
    ##
    return inputs_good, targets_good
end
##
function maxabs_acceptable_signals_blsub(
        input_data, target_data; σ_maxabs::Real=0.43)
    ##
    _scale = maximum(abs.(input_data))
    max_abs = map(
        signal -> maximum(abs.(signal)) / _scale, eachslice(input_data; dims=5))
    idxs_good = findall(ma -> ma ≤ σ_maxabs, max_abs)
    ##
    targets_good = target_data[:, :, idxs_good]
    inputs_good = input_data[:, :, :, :, idxs_good]
    return inputs_good, targets_good
end
