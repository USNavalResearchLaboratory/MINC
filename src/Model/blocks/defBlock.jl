"""
Adds ApproxLayer to the lifting DefConv if L != 0.
"""
## DEFINING BLOCK
struct DefBlock{C} <: Lux.AbstractExplicitContainerLayer{(:chain,)}
    chain::C
end
##
function DefBlock(T::Int, t::Int, ordG::Int, ch::Pair{Int, Int};
        L::Int=0, init_weight::Function=kaiming_normal,
        pad::NTuple{2, Int}=(0, 0), groups::Int=1)
    # Channels
    chs_in = ch[1]
    chs_out = ch[2]
    # Conv
    def_conv = DefConv((t,), ordG, chs_in => chs_out;
        init_weight=init_weight, pad=pad, groups=groups)
    if L == 0
        chain = Chain(def_conv)
    else
        approx_layer = ApproxLayer{Val(0)}(ordG)
        chain = Chain(def_conv, approx_layer)
    end
    return DefBlock(chain)
end
##
function (m::DefBlock)(
        x::AbstractArray{Float32, 4}, ps::ComponentArray, st::NamedTuple)
    return m.chain(x, ps, st)
end
