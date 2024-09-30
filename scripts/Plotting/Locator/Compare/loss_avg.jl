## LOSSES
function fig_loss_avg(
        run_paths_array::Vector{Vector{String}}; metric=:loss_md2e_iou_exc)
    ##
    loss_stats = map(
        run_paths -> losses_model(run_paths; metric=metric), run_paths_array)
    #
    loss_train_avg = map(loss_stat -> mean(loss_stat[1]), loss_stats)
    loss_test_avg = map(loss_stat -> mean(loss_stat[2]), loss_stats)
    #
    loss_train_avg_avg = mean(loss_train_avg)
    loss_test_avg_avg = mean(loss_test_avg)
    #
    cfg = load_object(run_paths_array[1][1] * "/cfg.jld2")
    epochs = 0:(cfg.infotime):((length(loss_test_avg[1]) - 1) * cfg.infotime)
    ##
    fig = Figure()
    ax = Makie.Axis(fig[1, 1]; xlabel="Epoch", ylabel="RMSE (mm)",
        title="Average Losses vs. Epoch", yscale=log10)
    #
    lin_test = lines!(ax, epochs, loss_test_avg_avg; label="Test", color=:black)
    lin_train = lines!(ax, epochs, loss_train_avg_avg;
        label="Train", linestyle=:dash, color=:black)
    axislegend(ax)
    ##
    return fig
end
function plot_loss_avg(run_paths_array::Vector{Vector{String}};
        metric=:loss_md4e_iou_exc, save_for_paper=false)
    ##
    _fig() = fig_loss_avg(run_paths_array; metric=metric)
    fig = with_theme(_fig, theme_aps(); figure_padding=(1, 1, 1, 0))
    ##
    fig_saver(fig, run_paths_array, "loss_avg_$(metric)";
        save_for_paper=save_for_paper)
    ##
    return nothing
end
