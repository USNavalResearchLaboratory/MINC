##
using DrWatson
@quickactivate "MINC"
include(srcdir("MINC.jl"))
##
using JLD2
using Statistics, StatsBase
using NearestNeighbors
using LinearAlgebra
using FFTW
using CairoMakie, MakiePublication
using FileIO
using ParameterSchedulers
using Lux
using LaTeXStrings, MathTeXEngine
using MLUtils
#
CairoMakie.activate!()
#
include("Locator/Locator.jl")
include("Detector/Detector.jl")
include("Utils/Utils.jl")
include("gfx.jl")
#
function gfx_paper(; save_for_paper=true)
    ## LOCATE
    folder_path_full = "Locate/set=AT3_ratio=80_t0=725_t1=4000_w0=180_w1=419"
    folder_path_skip = "Locate/set=AT3_ratio=80_t0=1600_t1=4000_w0=137_w1=462"
    folder_path_truncate = "Locate/set=AT3_ratio=80_t0=725_t1=2395_w0=68_w1=538"
    # Commented out because data too large to version on GitHub
    # folder_path_window = "Locate/set=AT3_ratio=80_t0=1600_t1=2395_w0=1_w1=980"
    #
    folder_paths = [folder_path_full, folder_path_skip,
        folder_path_truncate, folder_path_window]
    ## DETECT
    folder_path_detect = "set=AT3_ratio=20_t0=725_t1=4000_w0=180_w1=419"
    ##
    gfx_paper_locator(folder_path_full; save_for_paper=save_for_paper)
    gfx_paper_detector(folder_path_detect; save_for_paper=save_for_paper)
    ## Makes figures and prints results containing information included in text/tables.
    map(fp -> gfx_locator_results(fp; metric=:loss_md1e, save_for_paper=false),
        folder_paths)
    ##
    return nothing
end
#
function gfx_supp(; save_for_paper=:supp)
    ##
    folder_path = "Locate/set=AT3_ratio=80_t0=725_t1=4000_w0=180_w1=419"
    ##
    model_paths = readdir(
        projectdir("_research/runs_archive/" * folder_path); join=true)
    run_paths_array = map(
        model_path -> readdir(model_path; join=true), model_paths)
    run_path = readdir(model_paths[1]; join=true)[1]
    ##
    plot_blsub_adj(run_path; save_for_paper=save_for_paper)
    plot_spectra_adj(run_path; save_for_paper=save_for_paper)
    plot_spectra_bl_adj(run_path; save_for_paper=save_for_paper)
    plot_abs_adj(run_path; save_for_paper=save_for_paper)
    plot_norm_adj(run_path; save_for_paper=save_for_paper)
    ##
    gfx_locator_supp(folder_path; save_for_paper=save_for_paper)
    ##
    return nothing
end
#
function gfx_other_paper(; save_for_paper=false)
    ##
    folder_path = "Locate/set=AT3_ratio=80_t0=725_t1=4000_w0=180_w1=419"
    #
    model_paths = readdir(
        projectdir("_research/runs_archive/" * folder_path); join=true)
    run_paths_array = map(
        model_path -> readdir(model_path; join=true), model_paths)
    run_path = readdir(model_paths[1]; join=true)[1]
    ##
    plot_bl_adj(run_path; save_for_paper=save_for_paper)
    ##
    return nothing
end
#
