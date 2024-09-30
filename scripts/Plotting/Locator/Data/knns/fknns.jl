## INDIVIDUAL AVERAGES
function plot_fknn_cleaned(run_path::String; save_for_paper=false)
    ##
    cfg = load_object(run_path * "/cfg.jld2")
    # Process data config
    set = cfg.set
    #
    ω_min = cfg.ω_min
    ω_max = cfg.ω_max
    σ_fknn = cfg.σ_fknn
    σ_rknn = cfg.σ_rknn
    pro_config = (; set, ω_min, ω_max, σ_fknn, σ_rknn)
    ##
    _fig_fknn_cleaned() = fig_fknn_cleaned(pro_config)
    fig = with_theme(
        _fig_fknn_cleaned, theme_aps_2col(); figure_padding=(0, 1, 0, 0))
    ##
    fig_saver(fig, run_path, "fknn_cleaned"; save_for_paper=save_for_paper)
    ##
    return nothing
    #
end
function fig_fknn_cleaned(pro_config)
    ## Processed data
    data_dict, file = produce_or_load(
        MINC.get_processed_data, pro_config, datadir("data_pro"); tag=false)
    @unpack signals_damaged_compressed, signals_baseline_compressed, locations = data_dict
    ##
    signals_damaged_trafod = map(
        signal -> rfft(signal), signals_damaged_compressed)
    signals_baseline_trafod = map(
        signal -> rfft(signal), signals_baseline_compressed)
    #
    signals_damaged_trafod_density = map(s -> abs.(s), signals_damaged_trafod)
    signals_baseline_trafod_density = map(s -> abs.(s), signals_baseline_trafod)
    signals_trafod_density = cat(
        signals_damaged_trafod_density, signals_baseline_trafod_density; dims=3)
    #
    send_trafod_density = map(s -> stack(signals_trafod_density[s, s, :]), 1:4)
    ##
    dists = map(s -> knn_dists(s, send_trafod_density), 1:4) .* 100
    #
    c_inds = CartesianIndices((2, 2))
    q = [3, 1, 2, 4]
    #
    #
    x_min = minimum(stack(dists))
    x_max = maximum(stack(dists))
    bins = 60
    bins = range(x_min, x_max, bins)
    fig = Figure()
    for s in 1:4
        ax = Makie.Axis(
            fig[Tuple(c_inds[q[s]])...]; title="Transducer $(s)", titlesize=10)
        hist!(dists[s]; normalization=:probability, bins=bins)
        xlims!(ax, minimum(stack(dists)), 1.45)
        ylims!(ax, 0, 0.270)
        if s ∉ [3, 4]
            ax.xticklabelsvisible = false
        end
        if s ∉ [2, 3]
            ax.yticklabelsvisible = false
        end
    end
    ##
    sideinfo = Label(fig[1:2, 0], "Probability Fraction"; rotation=pi / 2)
    bottinfo = Label(fig[3, 1:2], "Percent Difference from Mean Signal")
    supertitle = Label(fig[0, 1:2], "Spectral Amplitudes Comparison")
    ##
    return fig
end
