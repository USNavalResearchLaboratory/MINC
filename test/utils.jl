## 
function get_dataloader(rng, data::Tuple; cfg=MINC.Config())
    ##
    ratio = cfg.ratio
    batch_size = cfg.batch_size
    partial = cfg.partial
    #
    data_train, data_test = splitobs(shuffleobs(data); at=ratio)
    #
    loader_train = DataLoader(data_train; batchsize=batch_size, buffer=true,
        shuffle=true, partial=partial, rng=Lux.replicate(rng))
    loader_test = DataLoader(data_test; batchsize=batch_size, buffer=true,
        shuffle=false, partial=true, rng=Lux.replicate(rng))
    return loader_train, loader_test
end
