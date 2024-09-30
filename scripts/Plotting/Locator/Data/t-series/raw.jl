##
function fig_baseline_edge_raw(run_path::String)
    ##
    _signals_damaged, _signals_baseline, _, _ = read_data_raw(run_path)
    ## read_data_raw gets _times too but i prefer the interval to be [0, 0.4]
    _times = range(0, 0.4f-3, 10_000)
    ##
    t_start = 0.0725
    t_end = 0.4
    window = window_time_raw(t_start, t_end)
    ##
    signals_baseline = map(
        signal -> MINC.delete_zero_mode(signal[window]), _signals_baseline) .*
                       1_000
    times = _times[window] .* 1_000
    # mean over baselines
    signals_baseline_avg = dropdims(mean(signals_baseline; dims=3); dims=3)
    ## edge and opposing paths
    idxs_opp = [(1, 3), (2, 4), (3, 1), (4, 2)]
    idxs_diag = [(1, 1), (2, 2), (3, 3), (4, 4)]
    idxs = [(r, s) for r in 1:4, s in 1:4]
    idxs_edge = setdiff(idxs, hcat(idxs_opp, idxs_diag))
    ##
    signal_baseline_avg_opp = mean(map(
        idxs -> signals_baseline_avg[CartesianIndex(idxs)], idxs_opp))
    signal_baseline_avg_edge = mean(map(
        idxs -> signals_baseline_avg[CartesianIndex(idxs)], idxs_edge))
    ##
    fig = Figure()
    ax = Makie.Axis(fig[1, 1]; title="Baseline Signal (Edge)",
        ylabel="Response (mV)", xlabel="Time (ms)")
    lines!(ax, times, signal_baseline_avg_edge; linewidth=1)
    ##
    return fig
end
##
function fig_baseline_diag_raw(run_path::String)
    ##
    _signals_damaged, _signals_baseline, _, _ = read_data_raw(run_path)
    ## read_data_raw gets _times too but i prefer the interval to be [0, 0.4]
    _times = range(0, 0.4f-3, 10_000)
    ##
    t_start = 0.0725
    t_end = 0.4
    window = window_time_raw(t_start, t_end)
    ##
    signals_baseline = map(
        signal -> MINC.delete_zero_mode(signal[window]), _signals_baseline) .*
                       1_000
    times = _times[window] .* 1_000
    # mean over baselines
    signals_baseline_avg = dropdims(mean(signals_baseline; dims=3); dims=3)
    ## edge and opposing paths
    idxs_opp = [(1, 3), (2, 4), (3, 1), (4, 2)]
    idxs_diag = [(1, 1), (2, 2), (3, 3), (4, 4)]
    idxs = [(r, s) for r in 1:4, s in 1:4]
    idxs_edge = setdiff(idxs, hcat(idxs_opp, idxs_diag))
    ##
    signal_baseline_avg_opp = mean(map(
        idxs -> signals_baseline_avg[CartesianIndex(idxs)], idxs_opp))
    signal_baseline_avg_edge = mean(map(
        idxs -> signals_baseline_avg[CartesianIndex(idxs)], idxs_edge))
    ##
    fig = Figure()
    ax = Makie.Axis(fig[1, 1]; title="Baseline Signal (Diagonal)",
        ylabel="Response (mV)", xlabel="Time (ms)")
    lines!(ax, times, signal_baseline_avg_opp; linewidth=1)
    ##
    return fig
end
function plot_signal_pair_raw(run_path::String; save_for_paper=false)
    ##
    _fig_edge() = fig_baseline_edge_raw(run_path)
    fig = with_theme(_fig_edge, theme_aps(); figure_padding=(0, 1, 0, 0))
    ##
    fig_saver(fig, run_path, "baseline_edge_raw"; save_for_paper=save_for_paper)
    ##
    _fig_diag() = fig_baseline_diag_raw(run_path)
    fig = with_theme(_fig_diag, theme_aps(); figure_padding=(0, 1, 0, 0))
    ##
    fig_saver(fig, run_path, "baseline_diag_raw"; save_for_paper=save_for_paper)
    ##
    return nothing
end
