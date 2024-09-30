## LOGGING
include("logging/report.jl")
include("logging/preds.jl")
include("logging/metrics.jl")
## TRAINER
include("trainer.jl")
## UTILS
include("utils/loss_st.jl")
include("utils/losses.jl")
include("utils/eval_metrics.jl")
include("utils/utils.jl")
## LOOPS
include("jobs/locate.jl")
include("jobs/detect.jl")
include("jobs/image.jl")
