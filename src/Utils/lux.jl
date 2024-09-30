# FROM LUX SOURCE: https://github.com/LuxDL/Lux.jl/blob/eb980bdc5cb806f497a8ecb1ae31dfd860baa728/src/utils.jl#L92
# Convolution
function _convfilter(rng::AbstractRNG, filter::NTuple{N, Integer},
        ch::Pair{<:Integer, <:Integer};
        init=glorot_uniform, groups=1) where {N}
    cin, cout = ch
    @assert cin % groups==0 "Input channel dimension must be divisible by groups."
    @assert cout % groups==0 "Output channel dimension must be divisible by groups."
    return init(rng::AbstractRNG, filter..., cin ÷ groups, cout)
end
##
_maybetuple_string(pad) = string(pad)
function _maybetuple_string(pad::Tuple)
    return all(==(pad[1]), pad) ? string(pad[1]) : string(pad)
end
## Type stable
@inline function _eachslice(x::AbstractArray, ::Val{dims}) where {dims}
    return [selectdim(x, dims, i) for i in axes(x, dims)]
end
@inline function _eachslice(
        x::GPUArraysCore.AnyGPUArray, ::Val{dims}) where {dims}
    return [__unview(selectdim(x, dims, i)) for i in axes(x, dims)]
end

@inline __unview(x::SubArray) = copy(x)
@inline __unview(x) = x

function ∇_eachslice(Δ_raw, x::AbstractArray, ::Val{dims}) where {dims}
    Δs = CRC.unthunk(Δ_raw)
    i1 = findfirst(Δ -> Δ isa AbstractArray, Δs)
    i1 === nothing && zero.(x)  # all slices are Zero!
    Δ = similar(x)
    for i in axes(x, dims)
        Δi = selectdim(Δ, dims, i)
        if Δi isa CRC.AbstractZero
            fill!(Δi, 0)
        else
            copyto!(Δi, Δs[i])
        end
    end
    return CRC.ProjectTo(x)(Δ)
end
