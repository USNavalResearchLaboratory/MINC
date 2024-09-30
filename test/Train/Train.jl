## LOOPS
include("locate.jl")
include("detect.jl")
include("image.jl")
#
function run_train_tests()
    run_locate_test()
    run_detect_test()
    run_image_test()
    return nothing
end
