##
function get_model_name(run_paths::Vector{String})
    return get_model_name(run_paths[1])
end
function get_model_name(run_path::String)
    #
    model_type = split(split(run_path, "/")[(end - 1)], "_")[1]
    if model_type == "AL0"
        model_name = "Exactly Equivariant"
    elseif model_type == "AL1"
        model_name = "Approximately Equivariant"
    elseif model_type == "Z"
        model_name = "Ordinary"
    else
        model_name = "TITLE GRAB ERROR"
    end
    #
    return model_name
end
##
function model_name_abbv(name::String)
    if name == "Exactly Equivariant"
        return "Exact."
    elseif name == "Approximately Equivariant"
        return "Apprx."
    elseif name == "Ordinary"
        return "Ordny."
    else
        model_name = "TITLE GRAB ERROR"
    end
end
