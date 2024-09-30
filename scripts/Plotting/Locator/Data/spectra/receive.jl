## RECEIVE
function plot_receive_spectra(run_path::String; save_for_paper=false)
    ##
    signals_damaged, signals_baseline, locations, times = read_data_raw(run_path)
    ##
    ω_min = 180
    ω_max = 420
    ω_min_compressed, ω_max_compressed = MINC.freq_compress(ω_min, ω_max)
    ω_modes = ω_min_compressed:ω_max_compressed
    ##
    signals_baseline_avg = dropdims(mean(signals_baseline; dims=3); dims=3)
    ##
    signals_baseline_trafod = map(
        signal -> abs.(rfft(signal)[ω_modes]), signals_baseline_avg)
    ##
    signals_density = map(
        signal -> signal ./ sum(signal), signals_baseline_trafod)
    ## Sender 1
    s = 1
    ##
    function _fig_receive_spectra()
        return fig_receive_spectra(s, signals_density, ω_modes, times)
    end
    fig = with_theme(_fig_receive_spectra, theme_aps())
    ##
    fig_saver(fig, run_path, "spectra_sender=$(s)_compressed";
        save_for_paper=save_for_paper)
    ##
    return nothing
end
##
function fig_receive_spectra(s::Int, signals_density, ω_modes, times)
    ##
    dt = mean(diff(times))
    freqs = rfftfreq(length(times), 1 / dt) ./ 10^3 #kHz
    ##
    c_inds = CartesianIndices((2, 2))
    q = [3, 1, 2, 4]
    #
    fig = Figure()
    for r in 1:4
        ax = Makie.Axis(fig[Tuple(c_inds[q[r]])...]; title="Transducer $(r)",
            titlesize=10, xlabelsize=10, ylabelsize=10)
        lines!(ax, freqs[ω_modes], signals_density[r, s])
        ylims!(ax, 0, 0.1)
        if r ∉ [3, 4]
            ax.xticklabelsvisible = false
        end
        if r ∉ [2, 3]
            ax.yticklabelsvisible = false
        end
    end
    #
    sideinfo = Label(fig[1:2, 0], "Spectral Density"; rotation=pi / 2)
    bottinfo = Label(fig[3, 1:2], "Frequency (kHz)")
    supertitle = Label(fig[0, 1:2], "Recorded Spectra")
    ##
    return fig
end
