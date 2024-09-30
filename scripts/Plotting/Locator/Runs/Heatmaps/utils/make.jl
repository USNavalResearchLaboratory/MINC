##
function make_hgrid(errors, ys)
    # Put a pixel of side length 5 mm at the center of each load face
    # and then attribute neighboring error at 5 mm intervals
    ## 57 = 25 * 2 + 1 + 3 * 2
    fgrid = zeros(Float32, 57, 57)
    ngrid = zeros(Float32, 57, 57)
    ys_grid = (ys .* 25) .+ 25 .+ 1 .+ 3
    ##
    for i in 1:size(ys, 2)
        err = errors[i]
        coord = round.(Int, ys_grid[:, i])
        x, y = coord
        for n in -3:1:3
            for m in -3:1:3
                coord_x = x + n
                coord_y = y + m
                fgrid[coord_x, coord_y] += err
                ngrid[coord_x, coord_y] += 1
            end
        end
    end
    hgrid = fgrid ./ (ngrid .+ 1.0f-6)
    ##
    return hgrid
end
##
