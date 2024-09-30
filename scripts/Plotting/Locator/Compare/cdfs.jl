## CDFS
function fig_compare_cdfs(run_paths_array::Vector{Vector{String}}; bins=100)
    ##
    cdfs_array = map(run_paths -> cdfs_model(run_paths), run_paths_array)
    ##
    min_err = 3.35
    max_err = 2 * 3.35
    bins = 15
    bin_range = range(min_err, max_err, bins)
    ##
    cdfs_array_range = map(
        cdfs -> map(cdf -> cdf.(bin_range), cdfs), cdfs_array)
    cdfs_array_range_avg = map(cdfs -> mean(cdfs), cdfs_array_range)
    cdfs_array_range_std = map(cdfs -> std(cdfs), cdfs_array_range)
    #
    _labels = map(rps -> get_model_name(rps), run_paths_array)
    ##
    fig = Figure()
    ax = Makie.Axis(fig[1, 1]; xlabel="Distance Error (mm)",
        title="Cumulative Distributions", ylabel="Probability")
    for i in 1:length(run_paths_array)
        errorbars!(bin_range, cdfs_array_range_avg[i],
            cdfs_array_range_std[i]; linewidth=1, whiskerwidth=7)
    end
    for i in 1:length(run_paths_array)
        scatter!(ax, bin_range, cdfs_array_range_avg[i]; label=_labels[i])
    end
    ##
    axislegend(ax; position=:rb)
    return fig
end
##
function plot_compare_cdfs(
        run_paths_array::Vector{Vector{String}}; save_for_paper=false)
    ##
    _fig() = fig_compare_cdfs(run_paths_array)
    fig = with_theme(_fig, theme_aps(); figure_padding=(0, 1, 0, 0))
    ##
    fig_saver(fig, run_paths_array, "cdf"; save_for_paper=save_for_paper)
    return nothing
end
