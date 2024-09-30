function fig_hist_model(run_paths::Vector{String})
    ##
    com_errs = map(rp -> get_saved_array(rp, "com_errors"), run_paths)
    ##
    errs_avg = map(errs -> mean(errs), com_errs)
    errs_avg_avg = round(mean(errs_avg); digits=3)
    errs_avg_std = round(std(errs_avg); digits=3)
    ##
    errs_std = map(errs -> std(errs), com_errs)
    errs_std_avg = round(mean(errs_std); digits=3)
    errs_std_std = round(std(errs_std); digits=3)
    ##
    errs_worst = map(errs -> maximum(errs), com_errs)
    errs_worst_avg = round(mean(errs_worst); digits=3)
    errs_worst_std = round(std(errs_worst); digits=3)
    ## Collect all errors
    errs = reshape(stack(com_errs), :)
    #
    max_err = round(maximum(errs); digits=3)
    min_err = round(minimum(errs); digits=3)
    ##
    # bins = range(0, max_err; step=3.35)
    bins = range(0, max_err; step=2.35)
    #
    model_name = get_model_name(run_paths)
    # Histogram
    fig = Figure()
    ax = Makie.Axis(fig[1, 1]; xlabel="Distance Error (mm)",
        ylabel="Probability Fraction", title="Test Perf. ($(model_name))")
    hist!(errs; bins=bins, normalization=:probability)
    h1 = hist!([1000]; label="Mean = $(errs_avg_avg) ± $(errs_avg_std) mm",
        color=:transparent)
    h2 = hist!([1000]; label="Variance = $(errs_std_avg) ± $(errs_std_std) mm",
        color=:transparent)
    h3 = hist!([1000]; label="Worst = $(errs_worst_avg) ± $(errs_worst_std) mm",
        color=:transparent)
    # xlims!(ax, 0, 5 * 3.35)
    xlims!(ax, 0, 7 * 2.35)
    ylims!(ax, 0, 0.8)
    axislegend(ax)
    ##
    return fig
end
##
function plot_hist_model(run_paths::Vector{String}; save_for_paper=false)
    ##
    _fig_hist_model() = fig_hist_model(run_paths)
    fig = with_theme(_fig_hist_model, theme_aps())
    ##
    model_type = split(split(run_paths[1], "/")[(end - 1)], "_")[1]
    ##
    fig_saver(
        fig, run_paths, "hist_$(model_type)"; save_for_paper=save_for_paper)
    ##
    return nothing
end
##
