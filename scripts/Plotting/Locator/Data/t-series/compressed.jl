##
function fig_signal_pair_compressed_windowed(
        run_path, t_start, t_end, title; r=1, s=3, b=1)
    ##
    signals_damaged, signals_baseline, _, _ = read_data_raw(run_path)
    _times = range(0, 0.4f-3, 10_000)
    signals_damaged_compressed, signals_baseline_compressed, _, ω_modes = read_data_compressed(run_path)
    ##
    T = size(signals_damaged[1], 1)
    Γ = length(signals_damaged_compressed[1])
    times_compressed = range(0, 0.4f-3, Γ) .* 1_000
    #
    cfg = load_object(run_path * "/cfg.jld2")
    t_start_compressed, t_end_compressed = MINC.time_compress(
        t_start, t_end, cfg)
    window_compressed = t_start_compressed:t_end_compressed
    #
    times_compressed = times_compressed[window_compressed]
    # Redimensionalize
    rescaler = (sqrt(T / Γ) * 2π) ./ 1_000
    #
    signal_damaged = signals_damaged_compressed[r, s, b] ./ rescaler
    signal_baseline = signals_baseline_compressed[r, s, b] ./ rescaler
    #
    fig = Figure()
    ax = Makie.Axis(fig[1, 1]; ylabel="Response (mV)",
        xlabel="Time (ms)", title=title, titlesize=12)
    damaged = lines!(ax, times_compressed, signal_damaged[window_compressed];
        label="Damaged", linewidth=1, color=MakiePublication.COLORS[1][2])
    baseline = scatter!(
        ax, times_compressed, signal_baseline[window_compressed];
        label="Baseline", markersize=3, color=MakiePublication.COLORS[1][1])
    #
    ylims!(ax, -12.0, 10.0)
    axislegend(ax; position=:rb)
    ##
    return fig
end
function plot_signal_pair_compressed_windowed(run_path; save_for_paper=false)
    ##
    t_start = 0.2
    t_end = 0.365
    title = "Baseline vs. Damaged (Diagonal Flight Path)"
    function _fig()
        return fig_signal_pair_compressed_windowed(
            run_path, t_start, t_end, title)
    end
    fig = with_theme(_fig, theme_aps_2col(; heightwidthratio=0.34);
        figure_padding=(0, 1, 0, 0))
    ##
    fig_saver(fig, run_path, "pair_compressed_windowed";
        save_for_paper=save_for_paper)
    ##
    return nothing
end
