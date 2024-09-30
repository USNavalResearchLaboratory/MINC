## REPRESENTATIONS
function get_defining_reps(ordG::Int)
    if ordG == 8
        ρ_def = load_object(projectdir("src/Utils/Reps/precomputed/ρ_dih_def.jld2"))
    elseif ordG == 4
        ρ_def = load_object(projectdir("src/Utils/Reps/precomputed/ρ_cyc_def.jld2"))
    end
    return ρ_def
end
##
function get_defining_perms(ordG::Int)
    ρ_def = get_defining_reps(ordG)
    def_indices = [stack([findall(x -> x == 1, col)[1]
                          for col in eachslice(ρ; dims=2)]) for ρ in ρ_def] #[g][σ]
    return stack(def_indices) #[σ, g]
end
