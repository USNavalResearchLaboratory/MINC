## PREDICTIONS (LOCATOR)
function save_preds(loader, model::Union{Locator, Imager}, ps, st; cfg=Config())
    ##
    rescaler = length_units(cfg)
    ## PREDS/TARGETS
    preds, targets = get_preds_targets(loader, model, ps, st; cfg=cfg)
    ##
    com_errors = rescaler .* map(md1e, preds, targets) |> cpu_device()
    ##
    wsave(projectdir(cfg.savepath * "arrays/com_errors.jld2"),
        @strdict com_errors)
    ## Histogram
    function _fig()
        fig = Figure()
        ax = Makie.Axis(fig[1, 1]; xlabel="Center of Mass Distance Error (mm)",
            ylabel="Count (Test Data)")
        hist!(com_errors)
        err_mean = round(mean(com_errors); digits=1)
        err_std = round(std(com_errors); digits=1)
        err_worst = round(maximum(com_errors); digits=1)
        h1 = hist!([10]; label="Mean = $(err_mean) mm", color=:transparent)
        h2 = hist!([10]; label="Variance = $(err_std) mm", color=:transparent)
        h3 = hist!([10]; label="Worst = $(err_worst) mm", color=:transparent)
        axislegend(ax)
        return fig
    end
    fig = with_theme(_fig, theme_web())
    wsave(projectdir(cfg.savepath * "figs/comhist.png"), fig)
    return nothing
end
##
function save_preds(loader, model, ps, st; cfg=Config())
    # Do nothing
end
## UTILS
function get_preds_targets(loader, model::Locator, ps, st; cfg=Config())
    ##
    dev = cfg.dev
    ##
    preds = Vector{AbstractArray}()
    targets = Vector{AbstractArray}()
    ##
    for (x, y) in loader
        ##
        x = x |> dev
        y = y |> dev
        ##
        ŷ, _ = Lux.apply(model, x, ps, st)
        append!(preds, eachslice(ŷ; dims=ndims(ŷ)))
        append!(targets, eachslice(y; dims=ndims(y)))
    end
    return preds, targets
end
function get_preds_targets(loader, model::Imager, ps, st; cfg=Config())
    ##
    dev = cfg.dev
    ##
    preds = Vector{AbstractArray}()
    targets = Vector{AbstractArray}()
    ##
    for (x, y) in loader
        ##
        x = x |> dev
        y = y |> dev
        ##
        ŷ, _ = Lux.apply(model, x, ps, st)
        ŷ_vec = img_to_com(ŷ; cfg=cfg)
        y_vec = img_to_com(y; cfg=cfg)
        append!(preds, eachslice(ŷ_vec; dims=ndims(ŷ_vec)))
        append!(targets, eachslice(y_vec; dims=ndims(y_vec)))
    end
    return preds, targets
end
