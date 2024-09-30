function plot_abs_adj(run_path::String; save_for_paper=false)
    ##
    signals_damaged, signals_baseline, _, _ = read_data_raw(run_path)
    ##
    signals = map(signal -> signal .- mean(signal), signals_damaged)
    signals = map(signal -> maximum(abs.(signal)), signals)
    #
    x_min = 0.008 * 1_000
    x_max = 0.018 * 1_000
    bins = range(x_min, x_max, 64)
    function _fig()
        fig = Figure()
        for r in 1:4
            for s in 1:4
                ax = Makie.Axis(
                    fig[r, s]; title="Receiver = $(r), Sender = $(s)",
                    titlesize=7)
                hist!(ax, abs.(signals[r, s, :]) .* 1_000;
                    normalization=:probability, bins=bins)
                xlims!(ax, x_min, x_max)
                ylims!(ax, 0, 1)
                if s != 1
                    hideydecorations!(ax; ticks=false, minorticks=false)
                end
                if r != 4
                    hidexdecorations!(ax; ticks=false, minorticks=false)
                end
            end
        end
        botinfo = Label(fig[5, 1:4], "Maximum Signal Amplitude (mV)")
        sideinfo = Label(fig[1:4, 0], "Probability Fraction"; rotation=π / 2)
        topinfo = Label(fig[0, 1:4],
            " Adjacency Matrix of Damaged Signal Maximum Amplitudes")
        return fig
    end
    fig = with_theme(_fig, theme_aps_2col())
    ##
    fig_saver(fig, run_path, "abs_adj"; save_for_paper=save_for_paper)
    ##
    return nothing
end