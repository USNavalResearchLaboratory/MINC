# BASE
include("layers/groupConv/defConv.jl")
include("layers/groupConv/regConv.jl")
include("layers/groupConv/approxLayer.jl")
include("layers/pooler.jl")
# HEADS
include("layers/heads/vector.jl")
include("layers/heads/twoScalars.jl")
include("layers/heads/twoNumbers.jl")
include("layers/heads/grid.jl")
include("layers/heads/numberField.jl")
# BLOCKS
include("blocks/defBlock.jl")
include("blocks/regBlock.jl")
include("blocks/convBlock.jl")
# STACKS
include("stacks/stackA.jl")
include("stacks/stackZ.jl")
# MODELS
include("models/locators.jl")
include("models/imagers.jl")
include("models/detectors.jl")
##
include("utils.jl")
