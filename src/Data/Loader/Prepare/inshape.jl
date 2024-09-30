## LOCATE
function get_inshape_locate(x_train_r::AbstractArray{Float32, 4},
        y_train_r::AbstractArray{Float32, 2},
        _x_test::AbstractArray{Float32, 4}, _y_test::AbstractArray{Float32, 2})
    """
    There's no fancy n or bl indices here
    """
    x_train = x_train_r
    y_train = y_train_r
    x_test = _x_test
    y_test = _y_test
    ##
    return x_train, y_train, x_test, y_test
end
function get_inshape_locate(x_train_r::AbstractArray{Float32, 5},
        y_train_r::AbstractArray{Float32, 3},
        _x_test::AbstractArray{Float32, 5}, _y_test::AbstractArray{Float32, 3})
    """
    Here, in addition to t, r, s, and b indices, we have also a bl
    index corresponding to baseline subraction.
    """
    # size(x_train) = (t, r, s, bl * b) OR size(x_train) = (t, r, s, n * b)
    x_train = reshape(x_train_r, size(x_train_r)[1:3]..., :)
    # size(y_train) = (i, bl * b) or size(y_train) = (i, n * b)
    y_train = reshape(y_train_r, size(y_train_r, 1), :)
    # size(x_test) = (t, r, s, bl, b)  OR size(x_test) = (t, r, s, n, b) 
    x_test = reshape(_x_test, size(_x_test)[1:3]..., :, size(_x_test, 5))
    # size(y_test) = (i, b)
    y_test = _y_test[:, 1, :] #[i, b]
    ##
    return x_train, y_train, x_test, y_test
end
function get_inshape_locate(x_train_r::AbstractArray{Float32, 6},
        y_train_r::AbstractArray{Float32, 4},
        _x_test::AbstractArray{Float32, 6}, _y_test::AbstractArray{Float32, 4})
    """
    Here, in addition to t, r, s, and b indices, we have also a bl
    index corresponding to baseline subraction.
    """
    # size(x_train) = (t, r, s, n * bl * b)
    x_train = reshape(x_train_r, size(x_train_r)[1:3]..., :)
    # size(y_train) = (i, n * bl * b)
    y_train = reshape(y_train_r, size(y_train_r, 1), :)
    # size(x_test) = (t, r, s, n * bl, b)
    x_test = reshape(_x_test, size(_x_test)[1:3]..., :, size(_x_test, 6))
    # size(y_test) = (i, b)
    y_test = _y_test[:, 1, 1, :] #[i, b]
    ##
    return x_train, y_train, x_test, y_test
end
## DETECT
function get_inshape_detect(
        x_train_r::AbstractArray{Float32, 4}, y_train_r::AbstractArray{Bool, 2},
        _x_test::AbstractArray{Float32, 4}, _y_test::AbstractArray{Bool, 2})
    """
    There's no fancy n or bl indices here
    """
    x_train = x_train_r
    y_train = y_train_r
    x_test = _x_test
    y_test = _y_test
    ##
    return x_train, y_train, x_test, y_test
end
