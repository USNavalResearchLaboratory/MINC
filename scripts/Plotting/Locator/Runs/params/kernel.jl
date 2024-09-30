##
function plot_k1_weights(run_paths::Vector{String}; save_for_paper=false)
    ##
    ps_fft_avg_array = map(rp -> k1_weights(rp), run_paths)
    ##
    ps_fft_avg_avg = mean(ps_fft_avg_array)
    ps_fft_avg_std = std(ps_fft_avg_array)
    ##
    Λ = size(ps_fft_avg_avg, 1)
    times = range(0, 0.4, 10_000)
    dt = mean(diff(times))
    freqs = rfftfreq(length(times), 1 / dt) ./ 10^3 #kHz
    #
    cfg = load_object(run_paths[1] * "/cfg.jld2")
    ω_min = cfg.ω_min
    ω_max = cfg.ω_max
    ω_min_compressed, ω_max_compressed = MINC.freq_compress(ω_min, ω_max)
    ω_modes = round.(Int, range(ω_min_compressed, ω_max_compressed, Λ))
    ##
    function fig_ps()
        y_lim = maximum(abs.(ps_fft_avg_avg))
        fig = Figure()
        for r in 1:4
            for s in 1:4
                ax = Makie.Axis(fig[r, s])
                lines!(ax, freqs[ω_modes], ps_fft_avg_avg[:, r, s])
                band!(ax, freqs[ω_modes],
                    ps_fft_avg_avg[:, r, s] .- ps_fft_avg_std[:, r, s],
                    ps_fft_avg_avg[:, r, s] .+ ps_fft_avg_std[:, r, s];
                    color=(:blue, 0.25))
                ylims!(ax, 0, y_lim)
                hidedecorations!(ax)
            end
        end
        sideinfo = Label(fig[1:4, 0],
            "Spectral Weight Value [0 - $(round(y_lim; digits = 1))] / (Receiver Index)";
            rotation=pi / 2)
        botinfo = Label(
            fig[5, 1:4], "Frequency [180 - 419 kHz] / (Sender Index)")
        topinfo = Label(fig[0, 1:4], "Layer 1 Kernel")
        return fig
    end
    fig = with_theme(fig_ps, theme_aps_2col())
    ##
    fig_saver(fig, run_paths, "layer1_weights"; save_for_paper=save_for_paper)
    ##
    return nothing
end
##
function k1_weights(run_path::String)
    ##
    model, ps, st, cfg = get_model_ps_st_cfg(run_path)
    ordG = cfg.ordG
    ## Layer 1
    ps_L1 = ps.layer_1.layer_1.layer_1.weight
    T = size(ps_L1, 1)
    chs = size(ps_L1, 3)
    ##
    ps_fft_r = abs.(rfft(ps_L1, 1))
    ps_fft = reshape(ps_fft_r, :, 4, 4, chs)
    ps_fft_avg = dropdims(mean(ps_fft; dims=4); dims=4)
    return ps_fft_avg
end
