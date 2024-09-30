## KNN DISCARD 
function discard_time_outliers(
        signals_damaged_compressed, signals_baseline_compressed,
        locations; σ_rknn::Real=1.65, N::Int=3)
    for i in 1:N #resample mean
        signals_damaged_compressed, locations = rknn_acceptable_signals(
            signals_damaged_compressed, locations; σ_rknn=σ_rknn)
    end
    return signals_damaged_compressed, signals_baseline_compressed, locations
end
##
function rknn_acceptable_signals(
        signals::AbstractArray{Vector{Float32}, 3}; σ_rknn::Real=5)
    outlier_inds = rknn_outlier_inds(signals; σ_rknn=σ_rknn)
    inds_in = 1:size(signals, 3)
    inds_out = setdiff(inds_in, outlier_inds)
    signals_acceptable = signals[:, :, inds_out]
    return signals_acceptable
end
function rknn_acceptable_signals(signals::AbstractArray{Vector{Float32}, 3},
        locations; σ_rknn::Real=1.65)
    outlier_inds = rknn_outlier_inds(signals; σ_rknn=σ_rknn)
    inds_in = 1:size(signals, 3)
    inds_out = setdiff(inds_in, outlier_inds)
    signals_acceptable = signals[:, :, inds_out]
    locations_acceptable = locations[inds_out]
    return signals_acceptable, locations_acceptable
end
##
function rknn_outlier_inds(
        signals::AbstractArray{Vector{Float32}, 3}; σ_rknn::Real=1.65)
    ##
    idxs_bad_all = []
    for s in 1:4
        append!(idxs_bad_all, rknn_outlier_inds(s, signals; σ_rknn=σ_rknn))
    end
    ##
    return unique(idxs_bad_all)
end
##
function rknn_outlier_inds(
        s, signals::AbstractArray{Vector{Float32}, 3}; σ_rknn::Real=5)
    ##
    sends = stack(signals[s, s, :])
    ##
    tree = BruteTree(sends)
    k = size(sends, 2)
    point = dropdims(mean(sends; dims=2); dims=2)
    idxs, dists = knn(tree, point, k, true)
    ##
    cutoff = σ_rknn
    ##
    idxs_bad = idxs[findall(d -> d ≥ cutoff, dists)]
    ##
    return idxs_bad
end
