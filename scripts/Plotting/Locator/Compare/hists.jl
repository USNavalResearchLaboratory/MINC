function fig_compare_hist(run_paths_array::Vector{Vector{String}})
    ##
    model_mean_errs = map(rps -> mean_errs_model(rps), run_paths_array)
    ##
    model_names = map(rps -> get_model_name(rps), run_paths_array)
    model_mean_errs_std = map(mean_errs -> std(mean_errs), model_mean_errs)
    ##
    fig = Figure()
    _step = 0.1 #minimum(model_mean_errs_std)
    err_min = 2.35 #minimum(stack(model_mean_errs))
    err_max = 3.25 #maximum(stack(model_mean_errs))
    bins = range(err_min - _step, err_max + _step; step=_step)
    for m in 1:length(model_mean_errs)
        ax = Makie.Axis(fig[m, 1])
        ax.yminorticksvisible = false
        hist!(ax, model_mean_errs[m]; bins=bins)
        xlims!(ax, err_min, err_max)
        ylims!(ax, 0, 5)
        ax2 = Makie.Axis(fig[m, 1]; ylabel=model_name_abbv(model_names[m]),
            yaxisposition=:right)
        hidedecorations!(ax2; label=false)
        if m != length(run_paths_array)
            ax.xticklabelsvisible = false
        end
    end
    #
    topinfo = Label(fig[0, 1], "Histograms over Initializations")
    botinfo = Label(fig[4, 1], "Mean Distance Error (mm)")
    sideinfo = Label(fig[1:3, 0], "Count"; rotation=pi / 2)
    ##
    return fig
end
##
function plot_compare_hist(
        run_paths_array::Vector{Vector{String}}; save_for_paper=false)
    ##
    _fig() = fig_compare_hist(run_paths_array)
    fig = with_theme(_fig, theme_aps())
    ##
    fig_saver(fig, run_paths_array, "mean_hist"; save_for_paper=save_for_paper)
    ##
    return nothing
end
##
function mean_errs_model(run_paths::Vector{String})
    ##
    com_errs = map(rp -> get_saved_array(rp, "com_errors"), run_paths)
    ##
    errs_avg = map(errs -> mean(errs), com_errs)
    ##
    return errs_avg
end
