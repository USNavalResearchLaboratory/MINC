##
function gfx_paper_locator(folder_path; save_for_paper=true)
    ##
    model_paths = readdir(
        projectdir("_research/runs_archive/" * folder_path); join=true)
    run_paths_array = map(
        model_path -> readdir(model_path; join=true), model_paths)
    run_path = readdir(model_paths[1]; join=true)[1]
    ##
    gfx_locator_data(run_path; save_for_paper=save_for_paper)
    ## RESULTS
    gfx_locator_results(run_paths_array; save_for_paper=save_for_paper)
    ##
    return nothing
end
#
function gfx_paper_detector(folder_path; save_for_paper=true)
    ##
    exp_paths_detect = readdir(
        projectdir("_research/runs_archive/Detect/" * folder_path); join=true)
    run_paths_array = map(
        exp_path -> readdir(exp_path; join=true), exp_paths_detect)
    ## RESULTS
    gfx_detector_results(run_paths_array; save_for_paper=save_for_paper)
    ##
    return nothing
end
##
