##
function fig_locations(run_path::String)
    ##
    input_train, locs_train, input_test, locs_test = input_target_split(run_path)
    ##
    x_train = locs_train[1, :]
    y_train = locs_train[2, :]
    x_test = locs_test[1, :]
    y_test = locs_test[2, :]
    ##
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; aspect=DataAspect())
    ax.title = "Training Targets"
    ax.xlabel = "Horizontal Coordinate (mm)"
    ax.ylabel = "Vertical Coordinate (mm)"
    train = scatter!(x_train, y_train; label="Train", markersize=4.0)
    ##
    return fig
end
##
function plot_locations(run_path::String; save_for_paper=false)
    ##
    _fig_locations() = fig_locations(run_path)
    fig = with_theme(_fig_locations, theme_aps(; heightwidthratio=1.0))
    ##
    fig_saver(fig, run_path, "locations_scatter"; save_for_paper=save_for_paper)
    ##
    return nothing
end
