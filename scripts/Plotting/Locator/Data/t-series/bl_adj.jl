function plot_bl_adj(run_path::String; save_for_paper=false)
    ##
    _, signals_baseline, _, _ = read_data_raw(run_path)
    ##
    signals = mean(signals_baseline; dims=3)
    signals = map(s -> s .- mean(s), signals)
    signals = map(s -> s[1900:end], signals) .* 1_000
    times = range(0, 0.4, 10_000)[1900:end] .* 10
    #
    y_lim = maximum(abs.(stack(signals)))
    #
    function _fig()
        fig = Figure()
        for r in 1:4
            for s in 1:4
                ax = Makie.Axis(
                    fig[r, s]; title="Receiver = $(r), Sender = $(s)",
                    titlesize=7)
                lines!(ax, times, signals[r, s]; linewidth=0.5)
                ylims!(ax, -y_lim, y_lim)
                if s != 1
                    hideydecorations!(ax; ticks=false, minorticks=false)
                end
                if r != 4
                    hidexdecorations!(ax; ticks=false, minorticks=false)
                end
            end
        end
        botinfo = Label(fig[5, 1:4], L"Time ($10^{-4}$ s)")
        sideinfo = Label(fig[1:4, 0], "Response (mV)"; rotation=π / 2)
        topinfo = Label(
            fig[0, 1:4], L"Adjacency Matrix $V_{rs}(t)$ of Baseline Signals")
        return fig
    end
    fig = with_theme(_fig, theme_aps_2col())
    ##
    fig_saver(fig, run_path, "bl_adj"; save_for_paper=save_for_paper)
    ##
    return nothing
end
