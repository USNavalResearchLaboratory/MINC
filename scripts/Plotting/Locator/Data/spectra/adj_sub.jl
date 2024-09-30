## RECEIVE
function plot_spectra_bl_adj(run_path::String; save_for_paper=false)
    ##
    signals_damaged, signals_baseline, locations, times = read_data_raw(run_path)
    signals_damaged = map(s -> s .- mean(s), signals_damaged)
    signals_baseline = map(s -> s .- mean(s), signals_baseline)
    ##
    signals_sub = mean(signals_baseline; dims=3) .-
                  mean(signals_damaged; dims=3)
    ##
    ω_min = 180
    ω_max = 420
    ω_min_compressed, ω_max_compressed = MINC.freq_compress(ω_min, ω_max)
    ω_modes = ω_min_compressed:ω_max_compressed
    ##
    signals_trafod = map(signal -> abs.(rfft(signal)[ω_modes]), signals_sub)
    signals = map(signal -> signal ./ sum(signal), signals_trafod)
    ##
    y_lim = maximum(stack(signals))
    dt = mean(diff(times))
    freqs = rfftfreq(length(times), 1 / dt) ./ 10^4
    #
    function _fig()
        fig = Figure()
        for r in 1:4
            for s in 1:4
                ax = Makie.Axis(
                    fig[r, s]; title="Receiver = $(r), Sender = $(s)",
                    titlesize=7)
                if r == s
                    lines!(ax, freqs[ω_modes], signals[r, s] .* 0; linewidth=1)
                else
                    lines!(ax, freqs[ω_modes], signals[r, s]; linewidth=1)
                end
                ylims!(ax, 0, 0.1)
                if r != 4
                    hidexdecorations!(ax; ticks=false, minorticks=false)
                end
                if s != 1
                    hideydecorations!(ax; ticks=false, minorticks=false)
                end
            end
        end
        botinfo = Label(fig[5, 1:4], L"Frequency ($10^4$ Hz)")
        sideinfo = Label(fig[1:4, 0], "Damage Index"; rotation=π / 2)
        topinfo = Label(fig[0, 1:4],
            " Adjacency Matrix of Baseline-Subtracted Spectral Densities")
        return fig
    end
    #
    fig = with_theme(_fig, theme_aps_2col())
    ##
    fig_saver(fig, run_path, "spectra_bl_adj"; save_for_paper=save_for_paper)
    ##
    return nothing
end
##
