## LOSSES
function fig_compare_losses(
        run_paths_array::Vector{Vector{String}}; metric=:loss_md2e_iou_exc)
    ##
    loss_stats = map(
        run_paths -> losses_model(run_paths; metric=metric), run_paths_array)
    #
    loss_test_avg = map(loss_stat -> mean(loss_stat[2]), loss_stats)
    ##
    _labels = map(rps -> get_model_name(rps), run_paths_array)
    #
    fig = Figure()
    ax = Makie.Axis(fig[1, 1]; xlabel="Epoch", ylabel="RMSE (mm)",
        title="Model Test Losses vs. Epoch")
    #
    cfg = load_object(run_paths_array[1][1] * "/cfg.jld2")
    epochs = 0:(cfg.infotime):((length(loss_test_avg[1]) - 1) * cfg.infotime)
    epoch_start_idx = 10
    epochs = epochs[epoch_start_idx:end]
    loss_test_avg = map(lta -> lta[epoch_start_idx:end], loss_test_avg)
    #
    for m in 1:length(loss_test_avg)
        lin_test = lines!(ax, epochs, loss_test_avg[m]; linewidth=1)
        sct_test = scatter!(ax, epochs, loss_test_avg[m]; label=_labels[m])
    end
    axislegend(ax)
    ##
    return fig
end
function plot_compare_losses(run_paths_array::Vector{Vector{String}};
        metric=:loss_md4e_iou_exc, save_for_paper=false)
    ##
    _fig() = fig_compare_losses(run_paths_array; metric=metric)
    fig = with_theme(_fig, theme_aps(); figure_padding=(1, 1, 1, 0))
    ##
    fig_saver(fig, run_paths_array, "loss_compare_$(metric)";
        save_for_paper=save_for_paper)
    ##
    return nothing
end
