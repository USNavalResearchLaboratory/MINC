## Imager
function eval_loss(
        loader, model::Imager, ps::ComponentArray, st::NamedTuple; cfg=Config())
    ## Device
    dev = cfg.dev
    ##
    rescaler = length_units(cfg)
    ##
    l_md1e = 0.0f0
    l_md2e_iou_exc = 0.0f0
    l_md4e_iou_exc = 0.0f0
    l_ssim = 0.0f0
    worst = 0.0f0
    ntot = 0
    for (x, y_img) in loader
        x = x |> dev
        y_img = y_img |> dev
        ##
        ŷ_img, _ = Lux.apply(model, x, ps, st)
        ##
        ŷ = img_to_com(ŷ_img; cfg=cfg)
        y = img_to_com(y_img; cfg=cfg)
        ##
        errs_md1e = map(md1e, _eachslice(ŷ, Val(2)), _eachslice(y, Val(2)))
        errs_md2e_iou_exc = map(
            md2e_iou_exc, _eachslice(ŷ, Val(2)), _eachslice(y, Val(2)))
        errs_md4e_iou_exc = map(
            md4e_iou_exc, _eachslice(ŷ, Val(2)), _eachslice(y, Val(2)))
        errs_ssim = _ssim(ŷ_img, y_img; grid_length=model.grid_length)
        ##
        l_md1e += mean(errs_md1e) * size(x)[end]
        l_md2e_iou_exc += mean(errs_md2e_iou_exc) * size(x)[end]
        l_md4e_iou_exc += mean(errs_md4e_iou_exc) * size(x)[end]
        l_ssim += mean(errs_ssim) * size(x)[end]
        if maximum(errs_md1e) > worst
            worst = maximum(errs_md1e)
        end
        ##
        ntot += size(x)[end]
    end
    loss_md1e = rescaler * (l_md1e / ntot) |> rounder
    loss_md2e_iou_exc = rescaler * sqrt(l_md2e_iou_exc / ntot) |> rounder
    loss_md4e_iou_exc = rescaler * sqrt(sqrt(l_md4e_iou_exc / ntot)) |> rounder
    worst = rescaler * worst |> rounder
    loss_ssim = l_ssim / ntot |> rounder
    metrics = Dict(:loss_ssim => loss_ssim, :loss_md1e => loss_md1e,
        :loss_md4e_iou_exc => loss_md4e_iou_exc,
        :loss_md2e_iou_exc => loss_md2e_iou_exc, :worst => worst)
    return metrics
end
## Locator
function eval_loss(loader, model::Locator, ps::ComponentArray,
        st::NamedTuple; cfg=Config())
    ## Device
    dev = cfg.dev
    ##
    rescaler = length_units(cfg)
    ##
    l_md1e = 0.0f0
    l_md2e_iou_exc = 0.0f0
    l_md4e_iou_exc = 0.0f0
    worst = 0.0f0
    ntot = 0
    for (x, y) in loader
        ##
        x = x |> dev
        y = y |> dev
        ##
        ŷ, _ = Lux.apply(model, x, ps, st)
        ##
        errs_md1e = map(md1e, _eachslice(ŷ, Val(2)), _eachslice(y, Val(2)))
        errs_md2e_iou_exc = map(
            md2e_iou_exc, _eachslice(ŷ, Val(2)), _eachslice(y, Val(2)))
        errs_md4e_iou_exc = map(
            md4e_iou_exc, _eachslice(ŷ, Val(2)), _eachslice(y, Val(2)))
        ##
        l_md1e += mean(errs_md1e) * size(x)[end]
        l_md2e_iou_exc += mean(errs_md2e_iou_exc) * size(x)[end]
        l_md4e_iou_exc += mean(errs_md4e_iou_exc) * size(x)[end]
        if maximum(errs_md1e) > worst
            worst = maximum(errs_md1e)
        end
        ##
        ntot += size(x)[end]
    end
    loss_md1e = rescaler * (l_md1e / ntot) |> rounder
    loss_md2e_iou_exc = rescaler * sqrt(l_md2e_iou_exc / ntot) |> rounder
    loss_md4e_iou_exc = rescaler * sqrt(sqrt(l_md4e_iou_exc / ntot)) |> rounder
    worst = rescaler * worst |> rounder
    metrics = Dict(
        :loss_md1e => loss_md1e, :loss_md4e_iou_exc => loss_md4e_iou_exc,
        :loss_md2e_iou_exc => loss_md2e_iou_exc, :worst => worst)
    return metrics
end
## Detector
function eval_loss(loader, model::Detector, ps::ComponentArray,
        st::NamedTuple; cfg=Config())
    ##
    dev = cfg.dev
    l = 0.0f0
    acc = 0.0f0
    ntot = 0
    ##
    for (x, y) in loader
        ##
        x = x |> dev
        y = y |> dev
        ##
        ŷ, _ = Lux.apply(model, x, ps, st)
        ##
        l += logitbinarycrossentropy(ŷ, y) * size(x)[end]
        acc += sum(onecold(ŷ |> cpu_device()) .== onecold(y |> cpu_device()))
        ##
        ntot += size(x)[end]
    end
    loss = (l / ntot) |> rounder
    accuracy = (acc / ntot) * 100 |> rounder
    metrics = Dict(:loss => loss, :accuracy => accuracy)
    return metrics
end
## Generic
function eval_loss(
        loader, model, ps::ComponentArray, st::NamedTuple; cfg=Config())
    ## Device
    dev = cfg.dev
    ##
    l_mse = 0.0f0
    ntot = 0
    for (x, y) in loader
        ##
        x = x |> dev
        y = y |> dev
        ##
        ŷ, _ = Lux.apply(model, x, ps, st)
        ##
        l_mse += mse(ŷ, y) * size(x)[end]
        ##
        ntot += size(x)[end]
    end
    loss_mse = (l_mse / ntot) |> rounder
    metrics = Dict(:loss_mse => loss_mse)
    return metrics
end
