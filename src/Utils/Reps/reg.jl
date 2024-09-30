## REPRESENTATIONS
function get_regular_reps(ordG::Int)
    if ordG == 8
        ρ_reg = load_object(projectdir("src/Utils/Reps/precomputed/ρ_dih_reg.jld2"))
    elseif ordG == 4
        ρ_reg = load_object(projectdir("src/Utils/Reps/precomputed/ρ_cyc_reg.jld2"))
    end
    return ρ_reg
end
##
function get_regular_perms(ordG::Int)
    ρ_reg = get_regular_reps(ordG)
    reg_indices = [stack([findall(x -> x == 1, col)[1]
                          for col in eachslice(inv(ρ); dims=2)]) for ρ in ρ_reg] #[g][σ]
    return stack(reg_indices) #[σ, g]
end
