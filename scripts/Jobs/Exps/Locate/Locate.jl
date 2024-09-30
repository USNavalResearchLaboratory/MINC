##
include("exps.jl")
##
function exp_locate(seed::Int, cfg_exp::NamedTuple; cfg=MINC.Config(),
        dev_id::Int=0, bug_test::Bool=false)
    ## Unpack
    @unpack ratio, t_start, t_end, ω_min, ω_max = cfg_exp
    ## Exp
    cfg.seed = seed
    ##
    cfg.ratio = ratio
    cfg.t_start = t_start
    cfg.t_end = t_end
    cfg.ω_min = ω_min
    cfg.ω_max = ω_max
    ##
    save_name_exp = get_save_name_exp(cfg_exp)
    exp_name = "Locate/set=$(cfg.set)_" * save_name_exp
    ##
    if bug_test
        cfg.epochs = 2
        exp_name = "Locate_warmup/set=$(cfg.set)_" * save_name_exp
    else
        exp_name = "Locate/set=$(cfg.set)_" * save_name_exp
    end
    ## Model AL1
    cfg.model_type = "A"
    cfg.A_L = 1
    cfg.savepath = "runs/$(exp_name)/AL1/seed=$(cfg.seed)/"
    MINC.locate(; cfg=cfg, dev_id=dev_id, bug_test=bug_test)
    ## Model Z
    cfg.model_type = "Z"
    cfg.savepath = "runs/$(exp_name)/Z/seed=$(cfg.seed)/"
    MINC.locate(; cfg=cfg, dev_id=dev_id, bug_test=bug_test)
    ## Model AL0
    cfg.model_type = "A"
    cfg.A_L = 0
    cfg.savepath = "runs/$(exp_name)/AL0/seed=$(cfg.seed)/"
    MINC.locate(; cfg=cfg, dev_id=dev_id, bug_test=bug_test)
    return nothing
end
###
