##
function cdfs_model(run_paths::Vector{String})
    ##
    com_errs = map(rp -> get_saved_array(rp, "com_errors"), run_paths)
    ##
    cfgs = map(rp -> load_object(rp * "/cfg.jld2"), run_paths)
    L_max = only(unique(map(cfg -> cfg.L_max, cfgs)))
    L_min = only(unique(map(cfg -> cfg.L_min, cfgs)))
    shift = (L_max + L_min) / 2
    rescaler = L_max - shift
    ##
    max_err = maximum(stack(com_errs))
    min_err = minimum(stack(com_errs))
    ##
    cdfs = map(dist -> ecdf(dist), com_errs)
    function cdfs_avg(x)
        return mean(map(cdf -> cdf(x), cdfs))
    end
    function cdfs_std(x)
        return std(map(cdf -> cdf(x), cdfs))
    end
    model_name = get_model_name(run_paths)
    for x in 3.35 .* collect(0:10)
        cdf_x_avg = cdfs_avg(x)
        cdf_x_std = cdfs_std(x)
        println("$(model_name): Percentile beneath $(x) mm: $(cdf_x_avg) ± $(cdf_x_std)")
    end
    ##
    return cdfs
end
