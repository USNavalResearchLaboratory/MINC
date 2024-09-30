## FULL
function exp_locate_full(seed, ratio; dev_id=0, bug_test=false)
    # Config
    cfg_exp = get_cfg_exp(ratio; exp_type=:Full)
    # Warmup
    exp_locate(seed, cfg_exp; dev_id=dev_id, bug_test=true)
    # Experiment
    exp_locate(seed, cfg_exp; dev_id=dev_id, bug_test=bug_test)
    return nothing
end
## TRUNCATE
function exp_locate_truncate(seed, ratio; dev_id=0, bug_test=false)
    # Config
    cfg_exp = get_cfg_exp(ratio; exp_type=:Truncate)
    # Warmup
    exp_locate(seed, cfg_exp; dev_id=dev_id, bug_test=true)
    # Experiment
    exp_locate(seed, cfg_exp; dev_id=dev_id, bug_test=bug_test)
    return nothing
end
## SKIP
function exp_locate_skip(seed, ratio; dev_id=0, bug_test=false)
    # Config
    cfg_exp = get_cfg_exp(ratio; exp_type=:Skip)
    # Warmup
    exp_locate(seed, cfg_exp; dev_id=dev_id, bug_test=true)
    # Experiment
    exp_locate(seed, cfg_exp; dev_id=dev_id, bug_test=bug_test)
    return nothing
end
## WINDOW
function exp_locate_window(seed, ratio; dev_id=0, bug_test=false)
    # Config
    cfg_exp = get_cfg_exp(ratio; exp_type=:Window)
    # Warmup
    exp_locate(seed, cfg_exp; dev_id=dev_id, bug_test=true)
    # Experiment
    exp_locate(seed, cfg_exp; dev_id=dev_id, bug_test=bug_test)
    return nothing
end
