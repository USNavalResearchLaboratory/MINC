#
include("loss_acc.jl")
include("accs.jl")
##
function gfx_detector_results(run_paths_array; save_for_paper=false)
    ##
    plot_loss_acc(run_paths_array; save_for_paper=save_for_paper)
    plot_compare_accs(run_paths_array; save_for_paper=save_for_paper)
    ##
    return nothing
end
