## BASELINE AUGMENT
function baseline_subtract(
        baseline_subtract::Bool, signals_damaged::Array{Vector{Float32}, 3},
        signals_baseline::Array{Vector{Float32}, 3},
        locations::Array{Vector{Float32}, 1})
    ##
    if baseline_subtract
        signals_damaged_blsub = stack(stack([sd .- sb
                                             for sb in eachslice(
            signals_baseline; dims=3),
        sd in eachslice(signals_damaged; dims=3)]))
        ##
        locations_blsub = stack([l
                                 for bl in eachslice(signals_baseline; dims=3),
        l in locations])
        ##
        # size(input_blsub) = (t, r, s, bl, b)
        # size(target_blsub) = (i, bl, b)
    else
        signals_damaged_blsub = stack(signals_damaged)
        locations_blsub = stack(locations)
    end
    return signals_damaged_blsub, locations_blsub
end
