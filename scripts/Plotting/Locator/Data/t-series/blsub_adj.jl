function plot_blsub_adj(run_path::String; save_for_paper=false)
    ##
    signals_damaged, signals_baseline, _, _ = read_data_raw(run_path)
    signals_damaged = map(s -> s .- mean(s), signals_damaged)
    signals_baseline = map(s -> s .- mean(s), signals_baseline)
    ##
    signals = mean(signals_baseline; dims=3) .- mean(signals_damaged; dims=3)
    signals = map(s -> s[1725:end], signals) .* 1_000
    times = range(0, 0.4, 10_000)[1725:end]
    #
    y_lim = 0.6
    #
    function _fig()
        fig = Figure()
        for r in 1:4
            for s in 1:4
                ax = Makie.Axis(
                    fig[r, s]; title="Receiver = $(r), Sender = $(s)",
                    titlesize=7)
                if r == s
                    lines!(ax, times, signals[r, s] .* 0; linewidth=0.5)
                else
                    lines!(ax, times, signals[r, s]; linewidth=0.5)
                end
                ylims!(ax, -y_lim, y_lim)
                if s != 1
                    hideydecorations!(ax; ticks=false, minorticks=false)
                end
                if r != 4
                    hidexdecorations!(ax; ticks=false, minorticks=false)
                end
            end
        end
        botinfo = Label(fig[5, 1:4], "Time (ms)")
        sideinfo = Label(fig[1:4, 0], "Damage Index (mV)"; rotation=π / 2)
        topinfo = Label(
            fig[0, 1:4], " Adjacency Matrix of Baseline-Subtracted Signals")
        return fig
    end
    fig = with_theme(_fig, theme_aps_2col())
    ##
    fig_saver(fig, run_path, "blsub_adj"; save_for_paper=save_for_paper)
    ##
    return nothing
end
