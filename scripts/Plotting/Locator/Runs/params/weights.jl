##
function plot_sym_weights(run_paths::Vector{String}; save_for_paper=false)
    ##
    ps_layers_models = stack(map(rp -> sym_weights(rp), run_paths); dims=3)
    ##
    names_element = ["e", "s_{13}", "r^2", "s_{24}", "r^3", "s_h", "r", "s_v"]
    #
    width_bin = 0.15
    #
    bins = range(
        minimum(ps_layers_models), maximum(ps_layers_models); step=width_bin)
    bins_L = reverse(range(
        1.0 - width_bin / 2, minimum(ps_layers_models); step=-width_bin))
    bins_R = range(
        1.0 + width_bin / 2, maximum(ps_layers_models); step=width_bin)
    bins = vcat(bins_L, bins_R)
    #
    function fig_layer_model()
        fig = Figure()
        for l in 1:6
            for g in 1:8
                ax = Makie.Axis(fig[g, l])
                ax.yticks = ([0.0, 0.5, 1.0], ["0", "", "1"])
                hist!(ax, reshape(ps_layers_models[g, l, :], :);
                    bins=bins, normalization=:probability)
                ylims!(ax, 0, 1.0)
                xlims!(ax, 0.0, 2.51)
                hidedecorations!(ax; label=false, ticklabels=false, ticks=false)
                if l != 1
                    hideydecorations!(ax; ticks=false, minorticks=false)
                end
                if g != 8
                    hidexdecorations!(ax; ticks=false, minorticks=false)
                end
                if g == 1
                    ax.title = "Layer $(l)"
                    ax.titlesize = 8
                end
                sideinfo = Label(
                    fig[g, 7], L"%$(names_element[g])"; fontsize=20)
            end
            topinfo = Label(fig[0, 1:6],
                L"Histograms of the Symmetry-Breaking Weight $\omega(g)$ Values")
            botinfo = Label(fig[9, 1:6], "Weight Values")
            sideinfo = Label(
                fig[1:8, 0], "Probability Fraction"; rotation=pi / 2)
        end
        return fig
    end
    fig = with_theme(fig_layer_model, theme_aps_2col())
    ##
    fig_saver(fig, run_paths, "sym_weights"; save_for_paper=save_for_paper)
    ##
    return nothing
end
##
function sym_weights(run_path::String)
    ##
    model, ps, st, cfg = get_model_ps_st_cfg(run_path)
    ##
    ordG = cfg.ordG
    ## Layer 6
    ps_L6 = ordG .* softmax(ps.layer_2.dense_in.layer_2.weight ./ ordG)
    # Layer 5
    ps_L5 = ordG .*
            softmax(ps.layer_1.layer_5.layer_1.layer_1.layer_2.weight ./ ordG)
    # Layer 4
    ps_L4 = ordG .*
            softmax(ps.layer_1.layer_4.layer_1.layer_1.layer_2.weight ./ ordG)
    # Layer 3
    ps_L3 = ordG .*
            softmax(ps.layer_1.layer_3.layer_1.layer_1.layer_2.weight ./ ordG)
    # Layer 2
    ps_L2 = ordG .*
            softmax(ps.layer_1.layer_2.layer_1.layer_1.layer_2.weight ./ ordG)
    ## Layer 1
    ps_L1 = ordG .* softmax(ps.layer_1.layer_1.layer_2.weight ./ ordG)
    ##
    ps_layers = cat(ps_L1, ps_L2, ps_L3, ps_L4, ps_L5, ps_L6; dims=2) #[g, layer]
    ##
    return ps_layers
end
