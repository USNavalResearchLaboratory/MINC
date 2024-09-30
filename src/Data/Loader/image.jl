## DATALOADER
"""
Test/Train treated asymmetrically:

    Each baseline subtracted train input targets the corresponding baseline subtracted train
        target > Each test input is the collection of baseline subtractions for a given signal; we
        average over each of these predictions to produce a test prediction for the location
        associated with the given signal

Comment on mutation:

    L_max and L_min stored in cfg provide the information for reconstructing dimensionful quantities.
        They are in the un-centered (raw) coordinate system and their values could change depending
        on if edge data is discarded in the (pre)processing stage.
"""
function get_image_dataloader!(cfg::Config, rng::AbstractRNG; bug_test=false)
    ## Dataloader config
    ratio = cfg.ratio
    batch_size = cfg.batch_size
    partial = cfg.partial
    ordG = cfg.ordG
    ##
    input_data, target_data, L_max, L_min = prepare_locate_data(cfg)
    ## Update cfg!
    cfg.L_max = L_max
    cfg.L_min = L_min
    ## Test/Train split
    (x_train_r, y_train_r), (_x_test, _y_test) = splitobs(
        shuffleobs(Lux.replicate(rng), (input_data, target_data)); at=ratio)
    ## Reshaping 
    x_train, y_train, x_test, y_test = get_inshape_locate(
        x_train_r, y_train_r, _x_test, _y_test)
    ##
    if bug_test
        x_train = x_train[:, :, :, 1:20]
        y_train = y_train[:, 1:20]
        x_test = x_test[:, :, :, :, 1:4]
        y_test = y_test[:, 1:4]
        batch_size = 4
    end
    ##
    p = cfg.p
    σ = cfg.σ
    grid_length = cfg.grid_length
    y_train_img = vec_to_img(y_train; p=p, σ=σ, grid_length=grid_length)
    y_test_img = vec_to_img(y_test; p=p, σ=σ, grid_length=grid_length)
    ##
    data_train = (x_train, y_train_img)
    data_test = (x_test, y_test_img)
    ##
    loader_train = DataLoader(data_train; batchsize=batch_size, buffer=true,
        shuffle=true, partial=partial, rng=Lux.replicate(rng))
    loader_test = DataLoader(data_test; batchsize=batch_size, buffer=true,
        shuffle=false, partial=true, rng=Lux.replicate(rng))
    ##
    return loader_train, loader_test, cfg
end
##
