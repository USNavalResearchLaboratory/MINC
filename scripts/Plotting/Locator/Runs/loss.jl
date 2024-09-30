##
function losses_model(run_paths::Vector{String}; metric=:loss_md4e_iou_exc)
    ##
    metrics_test_histories = map(
        rp -> load_object(rp * "/arrays/metrics_test_history.jld2"), run_paths)
    metrics_train_histories = map(
        rp -> load_object(rp * "/arrays/metrics_train_history.jld2"), run_paths)
    ##
    _loss_train_histories = map(mth -> mth[metric], metrics_train_histories)
    _loss_test_histories = map(mth -> mth[metric], metrics_test_histories)
    ##
    infotime_window = 1:(minimum(map(l -> length(l) - 1, _loss_test_histories)) + 1)
    ##
    loss_test_histories = map(lth -> lth[infotime_window], _loss_test_histories)
    loss_train_histories = map(
        lth -> lth[infotime_window], _loss_train_histories)
    ##
    loss_train_finals = map(lth -> lth[end], loss_train_histories)
    loss_test_finals = map(lth -> lth[end], loss_test_histories)
    ##
    loss_test_finals_avg = mean(loss_test_finals)
    loss_test_finals_std = std(loss_test_finals)
    ##
    loss_train_finals_avg = mean(loss_train_finals)
    loss_train_finals_std = std(loss_train_finals)
    ##
    gen_gaps = loss_test_finals .- loss_train_finals
    gen_gaps_avg = mean(gen_gaps)
    gen_gaps_std = std(gen_gaps)
    ##
    model_name = get_model_name(run_paths)
    println("$(model_name): Final $(metric) = $(loss_test_finals_avg) ± $(loss_test_finals_std) mm")
    println("$(model_name): Generalization Gap = $(gen_gaps_avg) ± $(gen_gaps_std) mm")
    ##
    return loss_train_histories, loss_test_histories
end
