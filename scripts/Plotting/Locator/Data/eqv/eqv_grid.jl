## 
function plot_eqv_err_input(run_path::String; save_for_paper=false)
    ##
    model, ps_array, st_array, cfg = get_model_ps_st_cfg(run_path)
    ##
    _data_input, _data_target, _, _ = MINC.prepare_locate_data(cfg)
    ##
    data_input = eachslice(_data_input; dims=(2, 3, 4, 5))
    data_target = eachslice(_data_target[:, 1, :]; dims=2)
    ##
    hgrids = map(g -> plot_eqv_err_input(data_input, data_target, g), 1:8)
    hgrids_mean = mean_over_nonzero(hgrids)
    ##
    function fig_eqv_input()
        #
        fig = Figure()
        ax = Makie.Axis(
            fig[1, 1]; title="Input Data Equivariance Error", titlesize=8)
        ax.xlabel = "Horizontal Coordinate (mm)"
        ax.ylabel = "Vertical Coordinate (mm)"
        ax.xticks = ([7, 18, 29, 40, 51], ["-100", "-50", "0", "50", "100"])
        ax.yticks = ([7, 18, 29, 40, 51], ["-100", "-50", "0", "50", "100"])
        hm_avg = heatmap!(hgrids_mean)
        Colorbar(fig[:, 2], hm_avg; label="Relative Error",
            ticks=([1.27, 1.30, 1.33], ["1.28", "1.30", "1.33"]))
        colsize!(fig.layout, 1, Aspect(1, 1.0))
        resize_to_layout!(fig)
        #
        return fig
    end
    fig = with_theme(fig_eqv_input, theme_aps())
    ##
    fig_saver(
        fig, run_path, "input_eqv_err_grid"; save_for_paper=save_for_paper)
    ##
    return nothing
end
##
function plot_eqv_err_input(data_input, data_target, g::Int)
    ##
    errs, targets = eqv_err_input(data_input, data_target, g)
    hgrid = make_hgrid(errs, stack(targets))
    ##
    return hgrid
end
##
function eqv_err_input(data_input, data_target, g::Int)
    ##
    g_vec = MINC.get_vec_rep(8)[g, :, :]
    g_def = invperm(MINC.get_defining_perms(8)[:, g])
    ##
    data_target_rotated = map(target -> g_vec * target, data_target)
    ##
    inds_related = map(
        vec -> findall(
            target_rotated -> target_rotated == vec, data_target_rotated),
        data_target)
    ##
    inds_nonempty = findall(
        i -> ~isempty(inds_related[i]), 1:length(inds_related))
    ##
    inds_related_has_partner = map(i -> only(inds_related[i]), inds_nonempty)
    data_input_has_partner = data_input[:, :, :, inds_nonempty]
    data_target_has_partner = data_target[inds_nonempty]
    ##
    data_input_related = data_input_has_partner[g_def, g_def, :, :]
    ## [t, r, s, bl, b]
    diffs_abs2 = abs2.(stack(data_input_has_partner .- data_input_related))
    ## [t, 1, 1, bl, b]
    _norm = (1 / 2) .*
            (sum(abs2.(stack(data_input_has_partner)); dims=(2, 3)) .+
             sum(abs2.(stack(data_input_related)); dims=(2, 3)))
    ## [t, r, s, bl, b]
    diffs_abs2_norm = diffs_abs2 ./ (_norm .+ 1.0f-6)
    ## [t, 1, 1, 1, b]
    diffs_rmse_avg = mean(sqrt.(sum(diffs_abs2_norm; dims=(2, 3))); dims=(1, 4))
    ##
    return reshape(diffs_rmse_avg, :), data_target_has_partner
end
##
