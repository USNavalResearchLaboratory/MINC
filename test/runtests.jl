using DrWatson, Test
using Zygote, Lux, LuxCUDA
using LinearAlgebra, ComponentArrays, Random
using MLUtils
@quickactivate "MINC"
# Here you include files using `srcdir`
include(srcdir("MINC.jl"))
## Forward/Backward/Equivariance/Trainer Tests
include("Model/Model.jl")
include("Train/Train.jl")
include("tests.jl")
include("utils.jl")
## Run test suite
println("Starting tests")
ti = time()

@testset "Locator tests" begin
    ## Locators
    @test test_model(LocatorATest()) == true
    @test test_model(LocatorZTest()) == true
    ##
end
@testset "Detector tests" begin
    ## Detectors
    @test test_model(DetectorATest()) == true
    @test test_model(DetectorZTest()) == true
    ##
end
@testset "layer tests" begin
    ## Heads
    @test test_model(ToVectorTest()) == true
    @test test_model(ToTwoScalarsTest()) == true
    @test test_model(ToTwoNumbersTest()) == true
    ## GroupConv
    @test test_model(DefConvTest()) == true
    @test test_model(RegConvTest()) == true
    @test test_model(ApproxLayerTest()) == true
end
@testset "block tests" begin
    ## Blocks
    @test test_model(DefBlockTest()) == true
    @test test_model(RegBlockTest()) == true
    @test test_model(ConvBlockTest()) == true
    ##
end
@testset "stack tests" begin
    ## Stacks
    @test test_model(StackATest()) == true
    @test test_model(StackZTest()) == true
    ##
end
@testset "Imager tests" begin
    ## Imagers
    @test test_model(ImagerATest()) == true
    @test test_model(ImagerZTest()) == true
    @test test_model(ToGridTest()) == true
    @test test_model(ToNumberFieldTest()) == true
    ##
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti / 60; digits=3), " minutes")
