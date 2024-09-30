##
function G_hgrids_flucts(hgrids::Vector{Matrix{Float32}})
    G_hgrids = stack(stack(map(hgrid -> get_rotated_hgrid(hgrid), hgrids)))
    G_hgrid_avgs = stack(map(
        g_hgrids -> mean_over_nonzero(eachslice(g_hgrids; dims=3)),
        eachslice(G_hgrids; dims=3)))
    G_hgrids_flucts = stack(map(
        G_hgrid -> G_hgrid .- G_hgrid_avgs, eachslice(G_hgrids; dims=4)))
    return G_hgrids_flucts
end
##
function CC(i, g_hgrid_flucts)
    return CC(i, 1, g_hgrid_flucts)
end
function CC(i, j, g_hgrid_flucts)
    C_i = g_hgrid_flucts[:, :, i, :] #[x, y, run]
    C_j = g_hgrid_flucts[:, :, j, :] #[x, y, run]
    CC_ij = dropdims(mean(C_i .* C_j; dims=3); dims=3)
    return CC_ij
end
## Rotating
function get_rotated_hgrid(hgrid)
    grid_inds = get_grid_inds(hgrid)
    hgrid = map(inds -> hgrid[inds], eachslice(grid_inds; dims=3))
    return hgrid
end
#
function get_grid_inds(hgrid)
    grid_length = size(hgrid, 1)
    if isodd(grid_length)
        return _get_grid_inds(grid_length)
    else
        throw(DomainError("$grid_length is not odd"))
    end
end
#
function _get_grid_inds(grid_length::Int64)
    #
    R = MINC.get_vec_rep(8)
    R_r = reshape(R, :, 2) #[m * r * i, j]
    #
    shift = div(grid_length, 2) + 1
    c_inds = CartesianIndices((grid_length, grid_length)) #cartesian_inds
    #
    v_inds = reverse(
        [reverse(collect(Tuple(c_ind))) .- shift for c_ind in c_inds]; dims=1) #vector_inds
    v_inds_r = stack(reshape(v_inds, :)) #[j, x * y]
    #
    R_r_v_inds_r = round.(Int, R_r * v_inds_r) #[m * r * i, j] x [j, x * y] -> [m * r * i, x * y]
    R_v_inds = reshape(R_r_v_inds_r, 8, 2, grid_length, grid_length) #[g, i, x, y]
    #
    es_R_v_inds = permutedims(eachslice(R_v_inds; dims=(1, 3, 4)), (2, 3, 1)) #[x, y, g][i]
    grid_inds = reverse(
        [CartesianIndex(Tuple(reverse(es_R_v_ind .+ shift)))
         for es_R_v_ind in es_R_v_inds];
        dims=1) #[x, y, g]
    #
    return grid_inds #[x, y, g]
end
