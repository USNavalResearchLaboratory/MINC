##
function plot_loss_acc(
        run_paths_array::Vector{Vector{String}}; save_for_paper=false)
    ##
    loss_acc_array = map(rps -> plot_loss_acc(rps), run_paths_array)
    acc_train_avg_array = map(laa -> laa[1], loss_acc_array)
    acc_test_avg_array = map(laa -> laa[2], loss_acc_array)
    loss_train_avg_array = map(laa -> laa[3], loss_acc_array)
    loss_test_avg_array = map(laa -> laa[4], loss_acc_array)
    epochs_array = map(laa -> laa[5], loss_acc_array)
    ##
    model_names = map(rps -> get_model_name(rps), run_paths_array)
    ##
    acc_min = minimum(stack(acc_train_avg_array)) / 1.01
    acc_max = maximum(stack(acc_test_avg_array)) * 1.01
    loss_min = minimum(exp.(stack(loss_train_avg_array))) / 1.1
    loss_max = maximum(exp.(stack(loss_test_avg_array))) * 1.1
    #
    function _fig()
        fig = Figure()
        for i in 1:length(run_paths_array)
            ax1 = Makie.Axis(fig[1, i]; xlabel="Epoch", ylabel="Accuracy",
                title=model_names[i], yticksmirrored=false, titlesize=10)
            ax2 = Makie.Axis(fig[1, i]; xlabel="Epoch", ylabel="Perplexity",
                yaxisposition=:right, yticksmirrored=false)
            hidespines!(ax2)
            hidexdecorations!(ax2)
            ylims!(ax1, acc_min, acc_max)
            ylims!(ax2, loss_min, loss_max)
            #
            if i != 1
                ax1.yticklabelsvisible = 0
                ax1.ylabelvisible = 0
            end
            if i != 3
                ax2.yticklabelsvisible = 0
                ax2.ylabelvisible = 0
            end
            #
            lin_acc_train = scatterlines!(
                ax1, epochs_array[i], acc_train_avg_array[i];
                label="Train", markersize=5, marker=:circle, linewidth=1)
            lin_acc_test = scatterlines!(
                ax1, epochs_array[i], acc_test_avg_array[i]; label="Test",
                markersize=5, marker=:dtriangle, linewidth=1)
            axislegend(ax1; position=:rc)
            #
            lin_loss_train = scatterlines!(
                ax2, epochs_array[i], exp.(loss_train_avg_array[i]);
                label="Train", linestyle=:dash,
                marker=:circle, markersize=5, linewidth=1)
            lin_loss_test = scatterlines!(
                ax2, epochs_array[i], exp.(loss_test_avg_array[i]);
                label="Test", linestyle=:dash,
                marker=:dtriangle, markersize=5, linewidth=1)
        end
        topinfo = Label(fig[0, 1:3], "Detector Training Convergence")
        return fig
    end
    fig = with_theme(_fig, theme_aps_2col(; heightwidthratio=0.34);
        figure_padding=(0, 0, 1, 0))
    ##
    return fig_saver(
        fig, run_paths_array, "loss_acc"; save_for_paper=save_for_paper)
    ##
end

function plot_loss_acc(
        run_paths::Vector{String}; save_for_paper=false, save_at_all=false)
    ##
    cfg = load_object(run_paths[1] * "/cfg.jld2")
    ratio_train = round(Int, cfg.ratio * 100)
    ratio_test = 100 - ratio_train
    ##
    metrics_test_histories = map(
        rp -> load_object(rp * "/arrays/metrics_test_history.jld2"), run_paths)
    metrics_train_histories = map(
        rp -> load_object(rp * "/arrays/metrics_train_history.jld2"), run_paths)
    ##
    loss_train_arrays = map(mth -> mth[:loss], metrics_train_histories)
    loss_test_arrays = map(mth -> mth[:loss], metrics_test_histories)
    acc_train_arrays = map(mth -> mth[:accuracy], metrics_train_histories)
    acc_test_arrays = map(mth -> mth[:accuracy], metrics_test_histories)
    ##
    acc_test_avg = mean(acc_test_arrays)
    acc_test_std = std(acc_test_arrays)
    acc_train_avg = mean(acc_train_arrays)
    acc_train_std = std(acc_train_arrays)
    loss_test_avg = mean(loss_test_arrays)
    loss_test_std = std(loss_test_arrays)
    loss_train_avg = mean(loss_train_arrays)
    loss_train_std = std(loss_train_arrays)
    ##
    ppl_test_end = exp.(map(lta -> lta[end], loss_test_arrays))
    ppl_test_end_avg = mean(ppl_test_end)
    ppl_test_end_std = std(ppl_test_end)
    ##
    acc_test_end = map(aca -> aca[end], acc_test_arrays)
    acc_test_end_avg = mean(acc_test_end)
    acc_test_end_std = std(acc_test_end)
    ##
    model_name = split(split(run_paths[1], "/")[(end - 1)], "_")[1]
    println("$(model_name) Final PPL = $(ppl_test_end_avg) ± $(ppl_test_end_std)")
    println("$(model_name) Final Acc = $(acc_test_end_avg) ± $(acc_test_end_std)")
    #
    acc_test_end_under_99 = length(acc_test_end[acc_test_end .< 99])
    println("$(model_name) Number Under 99 = $(acc_test_end_under_99)")
    println("$(model_name) Best Acc = $(maximum(acc_test_end))")
    println("$(model_name) (Best - Mean) Acc = $(maximum(acc_test_end) - acc_test_end_avg)")
    #
    println("$(model_name) Best PPL = $(minimum(ppl_test_end))")
    println("$(model_name) (Mean - Best) PPL = $(-minimum(ppl_test_end) + ppl_test_end_avg)")
    ##
    epochs = 0:(cfg.infotime):((length(loss_test_avg) - 1) * cfg.infotime)
    ##
    return acc_train_avg, acc_test_avg, loss_train_avg, loss_test_avg, epochs
end
