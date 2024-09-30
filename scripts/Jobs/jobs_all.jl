## 
using DrWatson
@quickactivate "MINC"
include(projectdir("scripts/Jobs/Jobs.jl"))
##
bug_test = false
##
dev_id = 0
seeds = [200, 300, 400, 500, 600, 700]
##
ratio_locate = 0.8
ratio_detect = 0.2
##
for seed in seeds
    ## DETECT
    exp_detect(seed, ratio_detect; dev_id=dev_id, bug_test=bug_test)
    ## LOCATE
    exp_locate_full(seed, ratio_locate; dev_id=dev_id, bug_test=bug_test)
    exp_locate_truncate(seed, ratio_locate; dev_id=dev_id, bug_test=bug_test)
    exp_locate_skip(seed, ratio_locate; dev_id=dev_id, bug_test=bug_test)
    # Commented out because data too large to version on GitHub
    # exp_locate_window(seed, ratio_locate; dev_id=dev_id, bug_test=bug_test)
end
