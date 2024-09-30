## 
function plot_eqv_err_bl(run_path::String; save_for_paper=false)
    ## 
    cfg = load_object(run_path * "/cfg.jld2")
    t_start, t_end = MINC.time_compress(cfg)
    ## Get data
    _, _signals_baseline, _, _ = read_data_compressed(run_path)
    signals_baseline = stack(_signals_baseline)[t_start:t_end, :, :, :]
    for r in 1:4
        for s in 1:4
            if r == s
                signals_baseline[:, r, s, :] .= 0
            end
        end
    end
    times = range(cfg.t_start, cfg.t_end, size(signals_baseline, 1))
    ## Mean Group Error [t, r, s, bl]
    diffs_rmse_avg_mean = mean(map(
        g -> eqv_err_bl(signals_baseline, g, run_path), 1:8))
    ##
    function fig_G_t()
        fig = Figure()
        ax = Makie.Axis(fig[1, 1])
        ax.title = "Baseline Signal Equivariance Error"
        ax.xlabel = "Time (ms)"
        ax.ylabel = "Relative Error"
        lines!(times, diffs_rmse_avg_mean; linewidth=1)
        return fig
    end
    fig = with_theme(fig_G_t, theme_aps_2col(; heightwidthratio=0.34);
        figure_padding=(1, 1, 0, 0))
    ##
    fig_saver(
        fig, run_path, "baseline_eqv_err_g_t"; save_for_paper=save_for_paper)
    ##
    return nothing
end
## 
function eqv_err_bl(
        signals_trafod::AbstractArray{Float32, 4}, g::Int, run_path::String)
    ## Get matrix representations
    g_inv_def = invperm(MINC.get_defining_perms(8)[:, g])
    ## Empty if related_vec location is discarded example
    #[t, r, s, bl]
    signals_trafod_related = signals_trafod[:, g_inv_def, g_inv_def, :]
    ##[t, r, s, bl]
    diffs_abs2 = abs2.(signals_trafod .- signals_trafod_related)
    ## Normalize pointwise in time (per bl)
    #[t, 1, 1, bl]
    _norm = (1 / 2) .* (sum(abs2.(signals_trafod); dims=(2, 3)) .+
             sum(abs2.(signals_trafod_related); dims=(2, 3)))
    ## [t, r, s, bl]
    diffs_abs2_norm = diffs_abs2 ./ (_norm .+ 1.0f-6)
    ## [t, 1, 1, bl]
    diffs_rmse = sqrt.(sum(diffs_abs2_norm; dims=(2, 3)))
    ## [t, 1, 1, 1]
    diffs_rmse_avg = mean(diffs_rmse; dims=4)
    ##
    return reshape(diffs_rmse_avg, :)
end
