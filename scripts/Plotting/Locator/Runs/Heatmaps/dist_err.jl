##
function dist_err_grids_test(run_paths::Vector{String}; save_for_paper=false)
    ## Load
    hgrids_test = map(rp -> heatgrid(rp; loader_type=:test), run_paths)
    ## Mean
    hgrid_test_mean = mean_over_nonzero(hgrids_test)
    ##
    model_name = get_model_name(run_paths)
    model_type = split(split(run_paths[1], "/")[(end - 1)], "_")[1]
    ## MEAN TEST (MANUAL COLORRANGE)
    colorrange = (0, 10)
    cfg_fig = (; title="Error Heatmap", cb_label="Mean Distance Error (mm)",
        titlesize=10, model_name=model_name)
    _fig_man() = heatgrid_fig(hgrid_test_mean, cfg_fig, colorrange)
    fig = with_theme(_fig_man, theme_aps())
    ##
    fig_saver(fig, run_paths, "heatgrid_test_$(model_type)_cr=manual";
        save_for_paper=save_for_paper)
    ## MEAN TEST (AUTO COLORRANGE)
    #cfg_fig = (;
    #    title="Error Heatmap", cb_label="Mean Distance Error (mm)", titlesize = 10,
    #    model_name = model_name)
    #_fig_auto() = heatgrid_fig(hgrid_test_mean, cfg_fig)
    #fig = with_theme(_fig_auto, theme_aps())
    ##
    #fig_saver(fig, run_paths, "heatgrid_test_$(model_type)_cr=auto";
    #    save_for_paper=save_for_paper)
    ##
    return nothing
end
##
function get_dist_err_grids_test(
        run_paths::Vector{String}; save_for_paper=false)
    ## Load
    hgrids_test = map(rp -> heatgrid(rp; loader_type=:test), run_paths)
    ## Mean
    hgrid_test_mean = mean_over_nonzero(hgrids_test)
    ##
    return hgrid_test_mean
end
