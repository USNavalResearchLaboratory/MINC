## DISCARD SPECTRAL OUTLIERS
function discard_spectral_outliers(
        signals_damaged_trafod, signals_baseline_trafod,
        locations; σ_fknn::Real=7, N::Int=5)
    ##
    for i in 1:N #resample mean
        signals_damaged_trafod, locations = fknn_acceptable_signals(
            signals_damaged_trafod, locations; σ_fknn=σ_fknn)
    end
    ##
    return signals_damaged_trafod, signals_baseline_trafod, locations
end
##
function fknn_acceptable_signals(signals_trafod::Array{Vector{ComplexF32}, 3},
        locations::Vector{Vector{Float32}}; σ_fknn::Real=7)
    ##
    outlier_inds = fknn_outlier_inds(signals_trafod; σ_fknn=σ_fknn)
    inds_in = 1:size(signals_trafod, 3)
    inds_out = setdiff(inds_in, outlier_inds)
    signals_trafod_acceptable = signals_trafod[:, :, inds_out]
    locations_acceptable = locations[inds_out]
    ##
    return signals_trafod_acceptable, locations_acceptable
end
function fknn_acceptable_signals(
        signals_baseline_trafod::Array{Vector{ComplexF32}, 3},
        signals_damaged_trafod::Array{Vector{ComplexF32}, 3}; σ_fknn::Real=7)
    ##
    outlier_inds = fknn_outlier_inds(
        signals_baseline_trafod, signals_damaged_trafod; σ_fknn=σ_fknn)
    inds_in = 1:size(signals_baseline_trafod, 3)
    inds_out = setdiff(inds_in, outlier_inds)
    signals_baseline_trafod_acceptable = signals_baseline_trafod[:, :, inds_out]
    ##
    return signals_baseline_trafod_acceptable
end
##
function fknn_outlier_inds(
        signals_trafod::Array{Vector{ComplexF32}, 3}; σ_fknn::Real=7)
    ##
    idxs_bad_all = Vector{Int}()
    for s in 1:4
        append!(
            idxs_bad_all, fknn_outlier_inds(s, signals_trafod; σ_fknn=σ_fknn))
    end
    ##
    return unique(idxs_bad_all)
end
function fknn_outlier_inds(
        signals_baseline_trafod::Array{Vector{ComplexF32}, 3},
        signals_damaged_trafod::Array{Vector{ComplexF32}, 3}; σ_fknn::Real=7)
    ##
    idxs_bad_all = Vector{Int}()
    for s in 1:4
        append!(idxs_bad_all,
            fknn_outlier_inds(s, signals_baseline_trafod,
                signals_damaged_trafod; σ_fknn=σ_fknn))
    end
    ##
    return unique(idxs_bad_all)
end
## START
function fknn_outlier_inds(
        s::Int, signals_trafod::AbstractArray{Vector{ComplexF32}, 3}; σ_fknn=7)
    ##
    signals_trafod_density = stack(map(
        signal -> abs.(signal) ./ norm(signal), signals_trafod[s, s, :]))
    ##
    tree = BruteTree(signals_trafod_density)
    k = size(signals_trafod_density, 2)
    point = dropdims(mean(signals_trafod_density; dims=2); dims=2)
    idxs, dists = knn(tree, point, k, true)
    ##
    dist_mean = mean(dists)
    dist_std = std(dists)
    cutoff = dist_mean + σ_fknn * dist_std
    ##
    idxs_bad = idxs[findall(d -> d ≥ cutoff, dists)]
    ##
    return idxs_bad
end
## END
function _fknn_outlier_inds(
        s::Int, signals_trafod::Array{Vector{ComplexF32}, 3}; σ_fknn=7)
    ##
    signals_trafod_density = stack(map(
        signal -> abs.(signal) ./ norm(signal), signals_trafod))
    signals_trafod_density_sends = stack(
        map(t -> signals_trafod_density[:, t, t, :], 1:4); dims=2)
    signals_trafod_density_s = signals_trafod_density_sends[:, s, :]
    ##
    tree = BruteTree(signals_trafod_density_s)
    k = size(signals_trafod_density_s, 2)
    point = dropdims(
        mean(signals_trafod_density_sends[:, s:s, :]; dims=(2, 3)); dims=(2, 3))
    idxs, dists = knn(tree, point, k, true)
    ##
    dist_mean = mean(dists)
    dist_std = std(dists)
    cutoff = dist_mean + σ_fknn * dist_std
    ##
    idxs_bad = idxs[findall(d -> d ≥ cutoff, dists)]
    idxs_good = idxs[findall(d -> d < cutoff, dists)]
    ##
    return idxs_bad
end
function _fknn_outlier_inds(
        s::Int, signals_baseline_trafod::Array{Vector{ComplexF32}, 3},
        signals_damaged_trafod::Array{Vector{ComplexF32}, 3}; σ_fknn=7)
    ##
    signals_damaged_trafod_density = stack(map(
        signal -> abs.(signal) ./ norm(signal), signals_damaged_trafod))
    signals_damaged_trafod_density_sends = stack(
        map(t -> signals_damaged_trafod_density[:, t, t, :], 1:4); dims=2)
    signals_baseline_trafod_density = stack(map(
        signal -> abs.(signal) ./ norm(signal), signals_baseline_trafod))
    signals_baseline_trafod_density_sends = stack(
        map(t -> signals_baseline_trafod_density[:, t, t, :], 1:4); dims=2)
    #
    signals_baseline_trafod_density_s = signals_baseline_trafod_density_sends[
        :, s, :]
    ##
    tree = BruteTree(signals_baseline_trafod_density_s)
    k = size(signals_baseline_trafod_density_s, 2)
    point = dropdims(
        mean(signals_damaged_trafod_density_sends[:, s:s, :]; dims=(2, 3));
        dims=(2, 3))
    idxs, dists = knn(tree, point, k, true)
    ##
    dist_mean = mean(dists)
    dist_std = std(dists)
    cutoff = dist_mean + σ_fknn * dist_std
    #
    idxs_bad = idxs[findall(d -> d ≥ cutoff, dists)]
    ##
    return idxs_bad
end
