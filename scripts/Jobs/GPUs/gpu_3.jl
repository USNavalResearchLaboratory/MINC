## GPU 3
using DrWatson
@quickactivate "MINC"
include(projectdir("scripts/Jobs/Jobs.jl"))
##
bug_test = false
##
dev_id = 3
seed = 300
##
ratio_locate = 0.8
ratio_detect = 0.2
## LOCATE
exp_locate_full(seed, ratio_locate; dev_id=dev_id, bug_test=bug_test)
exp_locate_truncate(seed, ratio_locate; dev_id=dev_id, bug_test=bug_test)
exp_locate_skip(seed, ratio_locate; dev_id=dev_id, bug_test=bug_test)
exp_locate_window(seed, ratio_locate; dev_id=dev_id, bug_test=bug_test)
## DETECT
exp_detect(seed, ratio_detect; dev_id=dev_id, bug_test=bug_test)
