##
function vis_abs()
    ##
    fig = Figure(; figsize=fig_size)
    ga = fig[1, 1:2] = GridLayout()
    gbl = fig[2, 1] = GridLayout()
    gbr = fig[2, 2] = GridLayout()
    gbl = fig[2, 2] = GridLayout()
    gbr = fig[2, 1] = GridLayout()
    #
    axtop = Makie.Axis(ga[1, 1:2]; ylabel="Response (mV)", yticklabelsize=12,
        xticklabelsize=12, ylabelsize=14, xlabelsize=14,
        xlabel="Time (ms)", title="Baseline vs. Damaged Signals")
    axbl = Makie.Axis(gbl[1, 1])
    axbr = Makie.Axis(gbr[1, 1]; xlabel="Epoch", ylabel="RMSE (mm)",
        title="Training Convergence", yticklabelsize=12,
        xticklabelsize=12, ylabelsize=14, xlabelsize=14)

    #
    hidedecorations!(axbl)
    ## LOSSES
    folder_path = "Locate/set=AT3_ratio=80_t0=725_t1=4000_w0=180_w1=419"
    model_paths = readdir(
        projectdir("_research/runs_archive/" * folder_path); join=true)
    run_paths_array = map(
        model_path -> readdir(model_path; join=true), model_paths)
    run_path = readdir(model_paths[1]; join=true)[1]
    #
    metric = :loss_md2e_iou_exc
    #
    loss_stats = map(
        run_paths -> losses_model(run_paths; metric=metric), run_paths_array)
    #
    loss_test_avg = map(loss_stat -> mean(loss_stat[2]), loss_stats)
    #
    _labels = map(rps -> get_model_name(rps), run_paths_array)
    #
    cfg = load_object(run_paths_array[1][1] * "/cfg.jld2")
    epochs = 0:(cfg.infotime):((length(loss_test_avg[1]) - 1) * cfg.infotime)
    epoch_start_idx = 10
    epochs = epochs[epoch_start_idx:end]
    loss_test_avg = map(lta -> lta[epoch_start_idx:end], loss_test_avg)
    #
    for m in 1:length(loss_test_avg)
        lin_test = lines!(axbr, epochs, loss_test_avg[m]; linewidth=1)
        sct_test = scatter!(axbr, epochs, loss_test_avg[m]; label=_labels[m])
    end
    axislegend(axbr; labelsize=12)
    ## PLATE
    plate = load("plate.png")
    plate = permutedims(plate, (2, 1))
    plate = reverse(plate; dims=2)
    image!(axbl, plate)
    hidespines!(axbl)
    axbl.aspect = size(plate)[1] / size(plate)[2]
    ## SIGNAL
    r = 1
    s = 3
    b = 1
    t_start = 0.2
    t_end = 0.365
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
    damaged = lines!(axtop, times_compressed, signal_damaged[window_compressed];
        label="Damaged", linewidth=1, color=MakiePublication.COLORS[1][2])
    baseline = scatter!(
        axtop, times_compressed, signal_baseline[window_compressed];
        label="Baseline", markersize=3, color=MakiePublication.COLORS[1][1])
    #
    ylims!(axtop, -12.0, 10.0)
    axislegend(axtop; position=:rb, labelsize=12)
    ##
    return fig
end
##
function plot_vis_abs()
    width = 8.0139 # in
    height = 6.2739 #in
    height_width_ratio = height / width
    fig_size = MakiePublication.figsize(width, height_width_ratio)
    #
    fig = with_theme(
        vis_abs, theme_aps(; width=width, heightwidthratio=height_width_ratio);
        figure_padding=(1, 1, 1, 0))
    ##
    # fig_saver(fig, run_paths_array, "loss_compare_$(metric)";
    #     save_for_paper=save_for_paper)
    return nothing
end
