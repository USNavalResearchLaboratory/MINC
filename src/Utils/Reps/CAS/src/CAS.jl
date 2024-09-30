#
## Heavy dependency
#
using JLD2
using Oscar
#
## REPRESENTATIONS
#
##
function get_defining_rep(σ::PermGroupElem)
    return permutedims(Float32.(Matrix(permutation_matrix(ZZ, σ))), (2, 1))
end
##
function get_defining_rep(G::PermGroup)
    ρ = map(σ -> get_defining_rep(σ), G)
    return (rep=ρ,)
end
##
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
    ρ = map(σ -> get_regular_rep(σ, G), G)
    return (rep=ρ,)
end
##
function save_ρs()
    ##
    ρ_def = get_defining_rep(dihedral_group(PermGroup, 8))
    ##
    save_object("rho_def.jld2", ρ_def)
    ##
    ρ_reg = get_regular_rep(dihedral_group(PermGroup, 8))
    ##
    save_object("rho_reg.jld2", ρ_reg)
    return nothing
end
