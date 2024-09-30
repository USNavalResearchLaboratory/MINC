##
function heatgrid_fig(hgrid::AbstractMatrix, cfg_fig::NamedTuple, colorrange)
    return _heatgrid_fig(hgrid, cfg_fig, colorrange)
end
function heatgrid_fig(hgrid::AbstractMatrix, cfg_fig::NamedTuple)
    colorrange = (minimum(hgrid), maximum(hgrid))
    return _heatgrid_fig(hgrid, cfg_fig, colorrange)
end
function _heatgrid_fig(hgrid::AbstractMatrix, cfg_fig::NamedTuple, colorrange)
    #
    fig = Figure()
    ax = Makie.Axis(fig[1, 1])
    ax.title = cfg_fig.title
    ax.titlesize = cfg_fig.titlesize
    ax.xlabel = "Horizontal Coordinate (mm)"
    ax.ylabel = "Vertical Coordinate (mm)"
    ax.xticks = ([7, 18, 29, 40, 51], ["-100", "-50", "0", "50", "100"])
    ax.yticks = ([7, 18, 29, 40, 51], ["-100", "-50", "0", "50", "100"])
    #
    hm = heatmap!(hgrid; colorrange=colorrange)
    if cfg_fig.model_name == "Approximately Equivariant"
        ax.ylabelvisible = false
        ax.yticklabelsvisible = false
    elseif cfg_fig.model_name == "Ordinary"
        Colorbar(fig[:, 2], hm; label=cfg_fig.cb_label)
        ax.ylabelvisible = false
        ax.yticklabelsvisible = false
    elseif cfg_fig.model_name == "Exactly Equivariant"
        # No colorbar
    else
        Colorbar(fig[:, 2], hm; label=cfg_fig.cb_label)
    end
    #
    colsize!(fig.layout, 1, Aspect(1, 1.0))
    resize_to_layout!(fig)
    #
    return fig
end
##
