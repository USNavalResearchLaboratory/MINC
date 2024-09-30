## FULL
function exp_detect(seed::Int, ratio::Real; dev_id=0, bug_test=false)
    # Config
    cfg_exp = get_cfg_exp(ratio; exp_type=:Full)
    # Warmup
    exp_detect(seed, cfg_exp; dev_id=dev_id, bug_test=true)
    # Experiment
    exp_detect(seed, cfg_exp; dev_id=dev_id, bug_test=bug_test)
    return nothing
end
