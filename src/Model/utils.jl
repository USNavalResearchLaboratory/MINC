##
function get_half_pad(T)
    if iseven(T)
        if iseven(div(T, 2, RoundDown))
            pad = (div(T, 4, RoundDown) - 1, div(T, 4, RoundDown))
        else
            pad = (div(T, 4, RoundDown), div(T, 4, RoundDown))
        end
    else
        if iseven(div(T - 1, 2, RoundDown))
            pad = (div(T - 1, 4, RoundDown), div(T - 1, 4, RoundDown))
        else
            pad = (div(T - 1, 4, RoundUp) - 1, div(T - 1, 4, RoundUp))
        end
    end
    return pad
end
##
