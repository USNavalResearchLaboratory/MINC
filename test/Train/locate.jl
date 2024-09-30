## LOCATOR
function run_locate_test(; cfg=MINC.Config(), dev_id=0, bug_test=true)
    ## CHEAP TEST
    cfg.epochs = 2
    cfg.infotime = 1
    ##
    cfg.model_type = "A"
    cfg.A_L = 1
    ## TRAIN 
    MINC.locate(; cfg=cfg, dev_id=dev_id, bug_test=bug_test)
    ##
    return true
end
