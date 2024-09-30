function fig_compare_accs(run_paths_array::Vector{Vector{String}})
    ##
    acc_tests_final_array = map(rps -> final_accs_model(rps), run_paths_array)
    acc_tests_final_std_array = map(atf -> std(atf), acc_tests_final_array)
    ##
    model_names = map(rps -> get_model_name(rps), run_paths_array)
    ##
    fig = Figure()
    _step = 0.1f0
    bin_min = 99.4f0
    bin_max = 100.0f0
    bins = range(bin_min, bin_max; step=_step)
    for m in 1:length(run_paths_array)
        ax = Makie.Axis(fig[m, 1]; title=model_names[m], titlesize=10)
        ax.xlabel = "Accuracy"
        ax.ylabel = "Count"
        ax.yticks = ([0, 1, 2, 3, 4], ["0", "", "2", "", "4"])
        ylims!(ax, 0, 5)
        ax.yminorticksvisible = false
        hist!(ax, acc_tests_final_array[m]; bins=bins)
        xlims!(ax, bin_min, bin_max)
        ax.xticks = ([99.6, 99.8, 100.0], ["0.996", "0.998", "1"])
        if m != length(run_paths_array)
            ax.xticklabelsvisible = false
            ax.xlabelvisible = false
        end
    end
    #
    topinfo = Label(
        fig[0, 1], "Accuracy Histograms over Initializations"; fontsize=12)
    ##
    return fig
end
##
function plot_compare_accs(
        run_paths_array::Vector{Vector{String}}; save_for_paper=false)
    ##
    _fig() = fig_compare_accs(run_paths_array)
    fig = with_theme(
        _fig, theme_aps(; heightwidthratio=1.0); figure_padding=(0, 0, 2, 0))
    ##
    fig_saver(fig, run_paths_array, "acc_hist"; save_for_paper=save_for_paper)
    ##
    return nothing
end
##
function final_accs_model(run_paths::Vector{String})
    ##
    metrics_test_histories = map(
        rp -> load_object(rp * "/arrays/metrics_test_history.jld2"), run_paths)
    ##
    acc_test_arrays = map(mth -> mth[:accuracy], metrics_test_histories)
    acc_test_final_array = map(ata -> ata[end], acc_test_arrays)
    ##
    return acc_test_final_array
end
##
