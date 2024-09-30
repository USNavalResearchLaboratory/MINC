## METRICS 
function save_metrics(epoch, metrics_train_history, metrics_test_history,
        model::Union{Detector, Locator, Imager}; cfg=Config())
    ##
    wsave(projectdir(cfg.savepath * "arrays/metrics_train_history.jld2"),
        @strdict metrics_train_history)
    wsave(projectdir(cfg.savepath * "arrays/metrics_test_history.jld2"),
        @strdict metrics_test_history)
    for key in keys(metrics_train_history)
        plot_metric(epoch, key, metrics_train_history[key],
            metrics_test_history[key], cfg.savepath; cfg=cfg)
    end
    ##
    return nothing
end
##
function save_metrics(
        epoch, metrics_train_history, metrics_test_history, model; cfg=Config())
    # For testing convenience
    # Don't plot when passing anything but a Model through trainer
end
##
function plot_metric(epoch, metric, metric_train_history,
        metric_test_history, savepath; cfg=Config())
    ##
    epochs = 0:(cfg.infotime):epoch
    ##
    function _fig()
        fig = Figure()
        if metric == :accuracy # no log10
            ax = Makie.Axis(fig[1, 1]; xlabel="Epoch", ylabel="$(metric)")
        else
            ax = Makie.Axis(
                fig[1, 1]; xlabel="Epoch", ylabel="$(metric)", yscale=log10)
        end
        lines!(ax, epochs, metric_train_history; label="Train", linestyle=:dash)
        lines!(ax, epochs, metric_test_history; label="Test")
        axislegend()
        return fig
    end
    ##
    fig = with_theme(_fig, theme_web())
    wsave(projectdir(savepath * "figs/$(metric)_history.png"), fig)
    ##
    return nothing
end
