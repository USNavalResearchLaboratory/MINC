##
include("Compare/Compare.jl")
include("Data/Data.jl")
include("Runs/Runs.jl")
include("Utils/Utils.jl")
##
## DATA
function gfx_locator_data(run_path::String; save_for_paper=false)
    ##
    plot_locations(run_path; save_for_paper=save_for_paper)
    plot_signal_pair_raw(run_path; save_for_paper=save_for_paper)
    plot_signal_pair_compressed_windowed(
        run_path; save_for_paper=save_for_paper)
    ##
    plot_fknn_cleaned(run_path; save_for_paper=save_for_paper)
    ##
    return nothing
end
## RESULTS
function gfx_locator_results(
        folder_path::String; save_for_paper=false, metric=:loss_md2e_iou_exc)
    ##
    model_paths = readdir(
        projectdir("_research/runs_archive/" * folder_path); join=true)
    run_paths_array = map(
        model_path -> readdir(model_path; join=true), model_paths)
    ##
    gfx_locator_results(
        run_paths_array; metric=metric, save_for_paper=save_for_paper)
    ##
    return nothing
end
function gfx_locator_results(run_paths_array::Vector{Vector{String}};
        save_for_paper=false, metric=:loss_md2e_iou_exc)
    ## Losses
    plot_compare_losses(
        run_paths_array; metric=metric, save_for_paper=save_for_paper)
    plot_loss_avg(run_paths_array; metric=metric, save_for_paper=save_for_paper)
    ## CDFs
    plot_compare_cdfs(run_paths_array; save_for_paper=save_for_paper)
    ##
    if ~save_for_paper
        ## Histogram (TEST)
        map(
            run_paths -> plot_hist_model(
                run_paths; save_for_paper=save_for_paper),
            run_paths_array)
        ## Compare Hist
        plot_compare_hist(run_paths_array; save_for_paper=save_for_paper)
    end
    ##
    return nothing
end
## SUPP MAT
function gfx_locator_supp(folder_path::String; save_for_paper=false)
    ##
    model_paths = readdir(
        projectdir("_research/runs_archive/" * folder_path); join=true)
    run_paths_array = map(
        model_path -> readdir(model_path; join=true), model_paths)
    ##
    gfx_locator_supp(run_paths_array; save_for_paper=save_for_paper)
    ##
    return nothing
end
function gfx_locator_supp(
        run_paths_array::Vector{Vector{String}}; save_for_paper=false)
    ##
    run_paths_AL1 = filter(rp -> occursin("AL1", rp), stack(run_paths_array))
    run_paths_Z = filter(rp -> occursin("Z", rp), stack(run_paths_array))
    run_path = run_paths_AL1[1]
    ## Symmetry Breaking Weights Values
    plot_sym_weights(run_paths_AL1; save_for_paper=save_for_paper)
    ## Data Equivariance
    plot_eqv_err_bl(run_path; save_for_paper=save_for_paper)
    plot_eqv_err_input(run_path; save_for_paper=save_for_paper)
    ## Distance Error Heatmaps (TEST)
    plot_dist_err_grids_test(run_paths_array; save_for_paper=save_for_paper)
    plot_dist_flucts_grids_test(run_paths_array; save_for_paper=save_for_paper)
    plot_eqv_err_grids_test(run_paths_array; save_for_paper=save_for_paper)
    ##
    return nothing
end
##
