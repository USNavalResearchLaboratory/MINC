##
function mean_over_nonzero(hgrids)
    ngrid = dropdims(
        sum(map(h -> h > 0 ? 1 : 0, stack(hgrids)); dims=3); dims=3)
    hgrid = sum(hgrids) ./ (ngrid .+ 1.0f-6)
    return hgrid
end
