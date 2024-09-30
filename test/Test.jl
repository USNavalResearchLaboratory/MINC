using DrWatson, Test
using Zygote, Lux, LuxCUDA
using LinearAlgebra, ComponentArrays
using MLUtils
using Statistics, Random
@quickactivate "MINC"
# Here you include files using `srcdir`
include(srcdir("MINC.jl"))
## Forward/Backward/Equivariance/Trainer Tests
include("Model/Model.jl")
## Training tests
include("Train/Train.jl")
# 
include("tests.jl")
include("utils.jl")
