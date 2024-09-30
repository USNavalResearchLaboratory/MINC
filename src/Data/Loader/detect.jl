## DATALOADER
function get_detect_dataloader(cfg::Config, rng::AbstractRNG; bug_test=false)
    ## Dataloader config
    ratio = cfg.ratio
    batch_size = cfg.batch_size
    partial = cfg.partial
    ordG = cfg.ordG
    ##
    input_data, target_data = prepare_detect_data(cfg)
    ## Test/Train split
    (x_train_r, y_train_r), (_x_test, _y_test) = splitobs(
        shuffleobs(Lux.replicate(rng), (input_data, target_data)); at=ratio)
    ## Reshaping 
    x_train, y_train, x_test, y_test = get_inshape_detect(
        x_train_r, y_train_r, _x_test, _y_test)
    ##
    if bug_test
        x_train = x_train[:, :, :, 1:12]
        y_train = y_train[:, 1:12]
        x_test = x_test[:, :, :, 1:2]
        y_test = y_test[:, 1:2]
        batch_size = 2
    end
    ##
    data_train = (x_train, y_train)
    data_test = (x_test, y_test)
    ##
    loader_train = DataLoader(data_train; batchsize=batch_size, buffer=true,
        shuffle=true, partial=partial, rng=Lux.replicate(rng))
    loader_test = DataLoader(data_test; batchsize=batch_size, buffer=true,
        shuffle=false, partial=true, rng=Lux.replicate(rng))
    ##
    return loader_train, loader_test, cfg
end
