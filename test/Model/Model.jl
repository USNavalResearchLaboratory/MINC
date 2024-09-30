## Abstract Types (for testing dispatch to check equivariance)
abstract type RepToRep end
abstract type TimeDefToTimeReg <: RepToRep end
abstract type TimeRegToTimeReg <: RepToRep end
abstract type TimeDefToVec <: RepToRep end
abstract type TimeRegToVec <: RepToRep end
abstract type TimeRegToBin <: RepToRep end
abstract type TimeDefToBin <: RepToRep end
abstract type TimeRegToGrid <: RepToRep end
abstract type TimeDefToGrid <: RepToRep end
## RToR types aren't checked for equivariance
abstract type RToR end
## BASE
include("layers/groupConv/defConv.jl")
include("layers/groupConv/regConv.jl")
include("layers/groupConv/approxLayer.jl")
## HEADS
include("layers/heads/vector.jl")
include("layers/heads/twoScalars.jl")
include("layers/heads/twoNumbers.jl")
include("layers/heads/grid.jl")
include("layers/heads/numberField.jl")
## BLOCKS
include("blocks/defBlock.jl")
include("blocks/regBlock.jl")
include("blocks/convBlock.jl")
## STACKS
include("stacks/stackA.jl")
include("stacks/stackZ.jl")
## LOCATORS
include("models/locators/locatorA.jl")
include("models/locators/locatorZ.jl")
## DETECTORS
include("models/detectors/detectorA.jl")
include("models/detectors/detectorZ.jl")
## IMAGERS
include("models/imagers/imagerA.jl")
include("models/imagers/imagerZ.jl")
##
LocatorTest = Union{LocatorATest, LocatorZTest}
DetectorTest = Union{DetectorATest, DetectorZTest}
##
