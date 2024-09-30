## Loss_st
function Loss_st(x, y, model, ps::ComponentArray, st::NamedTuple)
    ŷ, st_ = Lux.apply(model, x, ps, st)
    return mse(ŷ, y), st_
end
##
function Loss_st(x, y, model::Locator, ps::ComponentArray, st::NamedTuple)
    ŷ, st_ = Lux.apply(model, x, ps, st)
    return model.loss(ŷ, y), st_
end
##
function Loss_st(x, y, model::Detector, ps::ComponentArray, st::NamedTuple)
    ŷ, st_ = Lux.apply(model, x, ps, st)
    return logitbinarycrossentropy(ŷ, y), st_
end
function Loss_st(x, y, model::Imager, ps::ComponentArray, st::NamedTuple)
    ŷ, st_ = Lux.apply(model, x, ps, st)
    return _ssim(ŷ, y; grid_length=model.grid_length), st_
end
