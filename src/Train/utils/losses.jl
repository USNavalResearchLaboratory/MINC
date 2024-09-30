## SSIM
function _ssim(ŷ::AbstractArray{Float32, 3}, y::AbstractArray{Float32, 3};
        p::Int=2, σ::Float32=5.0f0, grid_length::Int=51)
    ##
    ŷ_r = reshape(ŷ, grid_length, grid_length, 1, :)
    y_r = reshape(y, grid_length, grid_length, 1, :)
    #
    return ssim_loss_fast(ŷ_r, y_r; crop=false)
end
## Losses
function mae(ŷ, y)
    return mean(abs.(ŷ .- y))
end
function mse(ŷ, y)
    return mean(abs2.(ŷ .- y))
end
function logitbinarycrossentropy(ŷ, y)
    return mean(@.((1 - y) * ŷ-logσ(ŷ)))
end
## Distance error
function md1e(ŷ::AbstractArray{Float32, 2}, y::AbstractArray{Float32, 2})
    dist2_err = sum(abs2.(ŷ .- y); dims=1)
    dist_err = mean(sqrt.(dist2_err))
    return dist_err
end
function md1e(ŷ::AbstractArray{Float32, 1}, y::AbstractArray{Float32, 1})
    dist_err = sqrt.(sum(abs2.(ŷ .- y)))
    return dist_err
end
## Distance squared error
function md2e(ŷ::AbstractArray{Float32, 2}, y::AbstractArray{Float32, 2})
    dist2_err = sum(abs2.(ŷ .- y); dims=1)
    return mean(dist2_err)
end
function md2e(ŷ::AbstractArray{Float32, 1}, y::AbstractArray{Float32, 1})
    dist2_err = sum(abs2.(ŷ .- y))
    return dist2_err
end
## Distance quartic error
function md4e(ŷ::AbstractArray{Float32, 2}, y::AbstractArray{Float32, 2})
    dist2_err = sum(abs2.(ŷ .- y); dims=1)
    dist4_err = mean(abs2.(dist2_err))
    return dist4_err
end
function md4e(ŷ::AbstractArray{Float32, 1}, y::AbstractArray{Float32, 1})
    dist2_err = sum(abs2.(ŷ .- y))
    dist4_err = abs2.(dist2_err)
    return dist4_err
end
## Intersection over union modifiers
function md2e_iou(ŷ::AbstractArray{Float32, 2}, y::AbstractArray{Float32, 2})
    ##
    dist2_err = sum(abs2.(ŷ .- y); dims=1)
    ##
    _iou_percentage = iou_percentage(ŷ, y)
    _md2e_iou = mean(dist2_err .* _iou_percentage)
    return _md2e_iou
end
function md2e_iou(ŷ::AbstractArray{Float32, 1}, y::AbstractArray{Float32, 1})
    ##
    dist2_err = sum(abs2.(ŷ .- y))
    ##
    _iou_percentage = iou_percentage(ŷ, y)
    _md2e_iou = dist2_err .* _iou_percentage
    return _md2e_iou
end
## Diffraction modifiers
function md2e_iou_exc(ŷ::AbstractArray, y::AbstractArray)
    ##
    dist_err_exc = dist_diffraction(ŷ, y)
    dist2_err_exc = abs2.(dist_err_exc)
    ##
    _iou_percentage = iou_percentage(ŷ, y)
    _md2e_iou_exc = mean(dist2_err_exc .* _iou_percentage)
    ##
    return _md2e_iou_exc
end
##
function md4e_iou_exc(ŷ::AbstractArray, y::AbstractArray)
    ##
    dist_err_exc = dist_diffraction(ŷ, y)
    dist2_err_exc = abs2.(dist_err_exc)
    dist4_err_exc = abs2.(dist2_err_exc)
    ##
    _iou_percentage = iou_percentage(ŷ, y)
    _md4e_iou_exc = mean(dist4_err_exc .* _iou_percentage)
    ##
    return _md4e_iou_exc
end
## UTILS
function iou_percentage(ŷ::AbstractArray, y::AbstractArray)
    #
    width_rescaled = Float32(40 / 125)
    coord_diffs = (ŷ .- y) ./ width_rescaled
    # xxx kludge for prod rrule failing on GPU when it encounters 0
    Ls = max.(1.0f-3, 1 .- abs.(coord_diffs))
    As = prod(Ls; dims=1)
    _iou_percentage = 1 .- As
    return _iou_percentage
end
##
function dist_diffraction(ŷ::AbstractArray, y::AbstractArray)
    #
    dist_err = md1e(ŷ, y)
    # xxx assumes particular wavelength of A0
    resolution = Float32(3.35f0 / 125.0f0)
    dist_err_exc = max.(dist_err, resolution)
    return dist_err_exc
end
