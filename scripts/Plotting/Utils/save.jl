## xxx typeof(save_for_paper) = Union{Bool, Symbol}
function fig_saver(fig, run_path::String, fname; save_for_paper=false)
    ## Puts the figure in folder_name
    if save_for_paper == :supp
        wsave(projectdir("docs/gfx_supp/$(fname).eps"), fig)
    elseif save_for_paper
        wsave(projectdir("docs/gfx/$(fname).eps"), fig)
    else
        _run_path = join(split(run_path, "/")[(end - 3):(end - 2)], "/")
        wsave(
            projectdir("_research/gfx_archive/" * _run_path * "/$(fname).png"),
            fig)
    end
    return nothing
end
function fig_saver(fig, run_paths::Vector{String}, fname; save_for_paper=false)
    if save_for_paper == :supp
        wsave(projectdir("docs/gfx_supp/$(fname).eps"), fig)
    elseif save_for_paper
        wsave(projectdir("docs/gfx/$(fname).eps"), fig)
    else
        _run_path = join(split(run_paths[1], "/")[(end - 3):(end - 1)], "/")
        wsave(
            projectdir("_research/gfx_archive/" * _run_path * "/$(fname).png"),
            fig)
    end
    return nothing
end
function fig_saver(fig, run_paths_array::Vector{Vector{String}},
        fname; save_for_paper=false)
    if save_for_paper == :supp
        wsave(projectdir("docs/gfx_supp/$(fname).eps"), fig)
    elseif save_for_paper
        wsave(projectdir("docs/gfx/$(fname).eps"), fig)
    else
        _run_path = join(
            split(run_paths_array[1][1], "/")[(end - 3):(end - 2)], "/")
        wsave(
            projectdir("_research/gfx_archive/" * _run_path * "/$(fname).png"),
            fig)
    end
    return nothing
end
##
