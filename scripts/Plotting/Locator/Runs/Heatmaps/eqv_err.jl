##
function eqv_err_grids_test(run_paths::Vector{String}; save_for_paper=false)
    ##
    model_name = get_model_name(run_paths)
    model_type = split(split(run_paths[1], "/")[(end - 1)], "_")[1]
    ## Load
    G_hgrids_test = map(
        g -> map(rp -> heatgrid_eqv(rp, g; loader_type=:test), run_paths), 1:8)
    names_element = ["e", "s_{13}", "r^2", "s_{24}", "r^3", "s_h", "r", "s_v"]
    ## MEAN TEST (MANUAL)
    colorrange = (5, 190)
    hgrid_test_mean = mean_over_nonzero(map(
        hgrids_test -> mean_over_nonzero(hgrids_test), G_hgrids_test))
    cfg_fig = (; title="Equivariance Error Heatmap",
        cb_label="Mean Error (mm)", titlesize=8, model_name="")
    _fig_man() = heatgrid_fig(hgrid_test_mean, cfg_fig, colorrange)
    fig = with_theme(_fig_man, theme_aps())
    ##
    fig_saver(fig, run_paths, "heatgrid_test_eqv_err_$(model_type)_cr=manual";
        save_for_paper=save_for_paper)
    ## MEAN TEST (AUTO)
    hgrid_test_mean = mean_over_nonzero(map(
        hgrids_test -> mean_over_nonzero(hgrids_test), G_hgrids_test))
    cfg_fig = (; title="Equivariance Error Heatmap",
        cb_label="Mean Error (mm)", titlesize=8, model_name="")
    _fig_auto() = heatgrid_fig(hgrid_test_mean, cfg_fig)
    fig = with_theme(_fig_auto, theme_aps())
    ##
    fig_saver(fig, run_paths, "heatgrid_test_eqv_err_$(model_type)_cr=auto";
        save_for_paper=save_for_paper)
    ##
    return nothing
end
##
function get_eqv_err_grids_test(run_paths::Vector{String}; save_for_paper=false)
    ##
    model_name = get_model_name(run_paths)
    model_type = split(split(run_paths[1], "/")[(end - 1)], "_")[1]
    ## Load
    G_hgrids_test = map(
        g -> map(rp -> heatgrid_eqv(rp, g; loader_type=:test), run_paths), 1:8)
    ##
    hgrid_test_mean = mean_over_nonzero(map(
        hgrids_test -> mean_over_nonzero(hgrids_test), G_hgrids_test))
    ##
    return hgrid_test_mean
end
