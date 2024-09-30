## FLUCTS
function get_hgrids_flucts(hgrids::Vector{Matrix{Float32}})
    hgrids_avg = mean(hgrids)
    hgrids_deltas = map(hgrid -> hgrid .- hgrids_avg, hgrids)
    hgrids_flucts = sqrt.(mean(map(
        (C1, C2) -> C1 .* C2, hgrids_deltas, hgrids_deltas)))
    return hgrids_flucts
end
