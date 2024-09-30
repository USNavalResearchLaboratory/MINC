##
function apply_g_to_x(x::AbstractArray{Float32, 5}, g::Vector{Int64})
    return x[:, g, g, :, :]
end
function apply_g_to_x(x::AbstractArray{Float32, 4}, g::Vector{Int64})
    return x[:, g, g, :]
end
function apply_g_to_y(y::AbstractArray{Float32, 2}, g::AbstractArray{Int64, 2})
    return g * y
end
##
function eqv_error_g(loader, model, ps, st, rescaler, g::Int)
    ##
    dev = gpu_device()
    ##
    ρ_vec = MINC.get_vec_rep(8) |> dev
    ρ_def = MINC.get_defining_perms(8)
    ##
    g_def = ρ_def[:, g]
    g_vec = ρ_vec[g, :, :]
    errors_g = Vector{}()
    for (x, y) in loader
        ##
        x = x |> dev
        y = y |> dev
        ##
        mx = model(x, ps, st)
        gmx = apply_g_to_y(mx, g_vec)
        gx = apply_g_to_x(x, g_def)
        mgx = model(gx, ps, st)
        ##
        errs = rescaler .*
               map(MINC.md1e, eachslice(gmx; dims=2), eachslice(mgx; dims=2))
        append!(errors_g, errs)
        ##
        println(mean(errs))
        ##
    end
    errors_g = errors_g |> cpu_device()
    return errors_g
end
