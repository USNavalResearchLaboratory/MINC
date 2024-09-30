## VEC CONNECTION
function get_vec_rep(ordG::Int)
    if ordG == 4
        return cyc_vec_rep()
    elseif ordG == 8
        return dih_vec_rep()
    end
end
"""
The basis I choose to represent the rotation matrices of the cyclic group on vectors
needs to be aligned (appropriately ordered) with the basis chosen for the defining/regular representations
coming from Oscar; this is achieved via aligning_perm.
"""
function cyc_vec_rep()
    R = zeros(Float32, 4, 2, 2) #[r, i, j]
    for r in 0:3
        m = 0
        R[r + 1, :, :] = [cos(r * π / 2)*(-1)^m sin(r * π / 2)*(-1)^(m + 1)
                          sin(r * π / 2) cos(r * π / 2)]
    end
    aligning_perm = [1, 3, 2, 4]
    R = R[aligning_perm, :, :]
    # 1s, 0s, -1s
    R = Int64.(round.(R))
    return R
end
##
function dih_vec_rep()
    R = zeros(Float32, 2, 4, 2, 2) #[m, r, i, j]
    for r in 0:3
        for m in 0:1
            R[m + 1, r + 1, :, :] = [cos(r * π / 2)*(-1)^m sin(r * π / 2)*(-1)^(m + 1)
                                     sin(r * π / 2) cos(r * π / 2)]
        end
    end
    R = reshape(R, :, 2, 2) #[m * r, i, j]
    aligning_perm = [1, 4, 5, 8, 3, 6, 7, 2]
    R = R[aligning_perm, :, :]
    # 1s, 0s, -1s
    R = Int64.(round.(R))
    return R
end
##
