## IMAGER
function run_image_test(; cfg=MINC.Config(), dev_id=0, bug_test=true)
    ## CHEAP TEST
    cfg.epochs = 2
    cfg.infotime = 1
    ##
    cfg.model_type = "A"
    cfg.A_L = 1
    ## TRAIN 
    MINC.image(; cfg=cfg, dev_id=dev_id, bug_test=bug_test)
    ##
    return true
end
