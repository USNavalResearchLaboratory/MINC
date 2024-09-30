##
function vec_to_img(vec::AbstractArray{Float32, 2}; p::Int=2,
        σ::Float32=5.0f0, grid_length::Int=51)
    ##
    _σ = σ / grid_length
    _range = collect(range(-1.0f0, 1.0f0; length=grid_length))
    _grid = reshape([(x, y) for x in _range, y in _range], :)
    ds_r = stack(
        map(_point -> dropdims(maximum(abs.(vec .- _point); dims=1); dims=1),
            _grid);
        dims=1)
    ##
    weight_field_r = softmax(_gauss.(ds_r; p=p, σ=_σ); dims=(1,))
    weight_field = reshape(weight_field_r, grid_length, grid_length, :)
    ##
    return weight_field
end
##
function _gauss(d::Float32; p::Int=2, σ::Float32=5.0f0)
    return exp(-(d^p) / (2(σ^p))) / Float32(sqrt(2 * π * (σ^p)))
end
##
function img_to_com(y_img::AbstractArray{Float32, 3}; cfg=Config())
    ##
    dev = cfg.dev
    ## xxx expresses coms in center-sup-norm convention
    grid_length = cfg.grid_length
    range_grid = collect(range(-1.0f0, 1.0f0; length=grid_length)) |> dev
    ##
    com_x = sum(
        sum(y_img; dims=2) .* reshape(range_grid, grid_length, 1); dims=1)
    com_y = sum(
        sum(y_img; dims=1) .* reshape(range_grid, 1, grid_length); dims=2)
    ##
    com = dropdims(hcat(com_x, com_y); dims=1)
    return com
end
