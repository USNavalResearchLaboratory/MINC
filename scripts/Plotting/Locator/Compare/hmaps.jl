## DISTANCE ERROR
function plot_dist_err_grids_test(
        run_paths_array::Vector{Vector{String}}; save_for_paper=false)
    ##
    hgrid_test_array = map(rps -> get_dist_err_grids_test(rps), run_paths_array)
    ##
    model_names = map(rps -> get_model_name(rps), run_paths_array)
    colorrange = (0, 10)
    cb_label = "Mean Distance Error (mm)"
    super_title = "Mean Distance Error Heatmaps"
    ##
    cfg_fig = (; model_names, colorrange, cb_label, super_title)
    ##
    _heatgrid_figs() = heatgrid_figs(hgrid_test_array, cfg_fig)
    fig = with_theme(_heatgrid_figs, theme_aps_2col(; heightwidthratio=0.34))
    ##
    return fig_saver(fig, run_paths_array, "heatgrid_dist_err_test";
        save_for_paper=save_for_paper)
    ##
    return nothing
end
## VARIANCE
function plot_dist_flucts_grids_test(
        run_paths_array::Vector{Vector{String}}; save_for_paper=false)
    ##
    hgrid_test_array = map(
        rps -> get_dist_flucts_grids_test(rps), run_paths_array)
    ##
    model_names = map(rps -> get_model_name(rps), run_paths_array)
    colorrange = (0, 5)
    cb_label = "Distance Error Variance (mm)"
    super_title = "Model Uncertainty Heatmaps"
    ##
    cfg_fig = (; model_names, colorrange, cb_label, super_title)
    ##
    _heatgrid_figs() = heatgrid_figs(hgrid_test_array, cfg_fig)
    fig = with_theme(_heatgrid_figs, theme_aps_2col(; heightwidthratio=0.34))
    ##
    return fig_saver(fig, run_paths_array, "heatgrid_dist_flucts_test";
        save_for_paper=save_for_paper)
    ##
    return nothing
end
## EQUIVARIANCE ERROR
function plot_eqv_err_grids_test(
        run_paths_array::Vector{Vector{String}}; save_for_paper=false)
    ##
    hgrid_test_array = map(rps -> get_eqv_err_grids_test(rps), run_paths_array)
    ##
    model_names = map(rps -> get_model_name(rps), run_paths_array)
    colorrange = (0, 190)
    cb_label = "Mean Error (mm)"
    super_title = "Equivariance Error Heatmaps"
    ##
    cfg_fig = (; model_names, colorrange, cb_label, super_title)
    ##
    _heatgrid_figs() = heatgrid_figs(hgrid_test_array, cfg_fig)
    fig = with_theme(_heatgrid_figs, theme_aps_2col(; heightwidthratio=0.34))
    ##
    return fig_saver(fig, run_paths_array, "heatgrid_eqv_err_test";
        save_for_paper=save_for_paper)
    ##
    return nothing
end
## FIGS
function heatgrid_figs(hgrid_test_array, cfg_fig)
    fig = Figure()
    for i in 1:length(hgrid_test_array)
        ax = Makie.Axis(fig[1, i]; title=cfg_fig.model_names[i],
            titlesize=8, aspect=DataAspect())
        hm = heatmap!(hgrid_test_array[i]; colorrange=cfg_fig.colorrange)
        ax.xlabel = "Horizontal Coordinate (mm)"
        ax.xticks = ([7, 18, 29, 40, 51], ["-100", "-50", "0", "50", "100"])
        ax.yticklabelsvisible = false
        if i == 1
            ax.ylabel = "Vertical Coordinate (mm)"
            ax.yticks = ([7, 18, 29, 40, 51], ["-100", "-50", "0", "50", "100"])
            ax.yticklabelsvisible = true
        elseif i == length(hgrid_test_array)
            Colorbar(fig[:, 4], hm; label=cfg_fig.cb_label)
            ax.ylabelvisible = false
            ax.yticklabelsvisible = false
        end
    end
    topinfo = Label(fig[0, 1:3], cfg_fig.super_title)
    return fig
end
