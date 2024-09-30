##
function dist_flucts_grids_test(run_paths::Vector{String}; save_for_paper=false)
    ## Load
    hgrids_test = map(rp -> heatgrid(rp; loader_type=:test), run_paths)
    ## Flucts
    hgrids_flucts = get_hgrids_flucts(hgrids_test)
    ##
    model_name = get_model_name(run_paths)
    model_type = split(split(run_paths[1], "/")[(end - 1)], "_")[1]
    ## MEAN TEST (MANUAL COLORRANGE)
    colorrange = (0, 5)
    cfg_fig = (;
        title="Uncertainty Heatmap", cb_label="Distance Error Variance (mm)",
        titlesize=10, model_name=model_name)
    _fig_man() = heatgrid_fig(hgrids_flucts, cfg_fig, colorrange)
    fig = with_theme(_fig_man, theme_aps())
    ##
    fig_saver(fig, run_paths, "heatgrid_flucts_test_$(model_type)_cr=manual";
        save_for_paper=save_for_paper)
    ## MEAN TEST (AUTO COLORRANGE)
    #cfg_fig = (;
    #    title="Uncertainty Heatmap", cb_label="Distance Error Variance (mm)", titlesize =
    #    10, model_name = model_name)
    #_fig_auto() = heatgrid_fig(hgrids_flucts, cfg_fig)
    #fig = with_theme(_fig_auto, theme_aps())
    ##
    #fig_saver(fig, run_paths, "heatgrid_flucts_test_$(model_type)_cr=auto";
    #    save_for_paper=save_for_paper)
    ##
    return nothing
end
##
function get_dist_flucts_grids_test(
        run_paths::Vector{String}; save_for_paper=false)
    ## Load
    hgrids_test = map(rp -> heatgrid(rp; loader_type=:test), run_paths)
    ## Flucts
    hgrids_flucts = get_hgrids_flucts(hgrids_test)
    ##
    return hgrids_flucts
end
