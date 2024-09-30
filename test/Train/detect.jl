## DETECTOR
function run_detect_test(; cfg=MINC.Config(), dev_id=1, bug_test=true)
    ## CHEAP TEST
    cfg.epochs = 2
    cfg.infotime = 1
    ##
    cfg.model_type = "A"
    cfg.A_L = 1
    ## TRAIN 
    min_loss = MINC.detect(; cfg=cfg, dev_id=dev_id, bug_test=bug_test)
    ##
    return true
end
