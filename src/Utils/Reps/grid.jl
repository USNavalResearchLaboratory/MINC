## GRID
function get_grid_rep(grid_length::Int64; ordG::Int64=8)
    ##
    @assert isodd(grid_length)
    ##
    R = get_vec_rep(ordG) #[g, i, j]
    R_r = reshape(R, :, 2) #g * i, j]
    ##
    shift = div(grid_length, 2) + 1
    ## Cartesian Inds
    c_inds = CartesianIndices((grid_length, grid_length)) #cartesian_inds
    ## Vector Inds
    v_inds = reverse(
        [reverse(collect(Tuple(c_ind))) .- shift for c_ind in c_inds]; dims=1)
    ##
    v_inds_r = stack(reshape(v_inds, :)) #[j, x * y]
    ## [g * i, j] x [j, x * y] -> [g * i, x * y]
    R_r_v_inds_r = round.(Int, R_r * v_inds_r)
    ##
    R_v_inds = reshape(R_r_v_inds_r, ordG, 2, grid_length, grid_length) #[g, i, x, y]
    R_v_inds_p = permutedims(R_v_inds, (1, 3, 4, 2)) #[g, x, y, i]
    ## [g][x, y][i]
    R_c_inds = eachslice(
        reverse(map(v_ind -> CartesianIndex(Tuple(reverse(v_ind .+ shift))),
            eachslice(R_v_inds_p; dims=(1, 2, 3))));
        dims=1)
    ##
    R_inds = stack(Tuple.(stack(R_c_inds))) #[i, x, y, g]
    aligning_perm = [1, 2, 3, 4, 5, 8, 7, 6]
    R_inds = R_inds[:, :, :, aligning_perm]
    ##
    return R_inds
end
