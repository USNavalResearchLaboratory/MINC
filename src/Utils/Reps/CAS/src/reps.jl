#
## Heavy dependency
#
using Oscar
using JLD2
##
function get_defining_rep(σ::PermGroupElem)
    return permutedims(Float32.(Matrix(permutation_matrix(ZZ, σ))), (2, 1))
end
##
function get_defining_rep(G::PermGroup)
    ρ = map(σ -> Bool.(get_defining_rep(σ)), G)
    return ρ
end
##
function get_regular_rep(σ::PermGroupElem, G::PermGroup)
    induced_perm = get_regular_perm(σ, G)
    return permutedims(
        Float32.(Matrix(permutation_matrix(ZZ, induced_perm))), (2, 1))
end
##
function get_regular_perm(σ::PermGroupElem, G::PermGroup)
    σG = map(g -> σ * g, G)
    σ_induced = map(σg -> first(findall(x -> x == σg, [g for g in G])), σG)
    return perm(σ_induced)
end
##
function get_regular_rep(G::PermGroup)
    return map(σ -> Bool.(get_regular_rep(σ, G)), G)
end
## 
function save_cyc_ρs()
    ##
    G = cyclic_group(PermGroup, 4)
    ρ_cyc_def = get_defining_rep(G)
    ρ_cyc_reg = get_regular_rep(G)
    ##
    jldsave("ρ_cyc_def.jld2"; ρ_cyc_def)
    jldsave("ρ_cyc_reg.jld2"; ρ_cyc_reg)
    ##
    return nothing
end
##
function save_dih_ρs()
    ##
    G = dihedral_group(PermGroup, 8)
    ρ_dih_def = get_defining_rep(G)
    ρ_dih_reg = get_regular_rep(G)
    ##
    jldsave("ρ_dih_def.jld2"; ρ_dih_def)
    jldsave("ρ_dih_reg.jld2"; ρ_dih_reg)
    ##
    return nothing
end
