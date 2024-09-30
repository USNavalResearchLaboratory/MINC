## WEIGHT AND LOCATION
function read_weight_and_locations(sender_folder_paths::Vector{String})
    ## Extract (weight, x, y) from naming conventions
    sender_is_1_csv_paths = readdir(sender_folder_paths[4]; join=true)
    sender_is_1_csv_names = readdir(sender_folder_paths[4]; join=false)
    ##
    weight_and_locations = [occursin("baseline", name) ? (0.0f0, 0.0f0, 0.0f0) :
                            parse.(Float32,
                                (split(name, "_")[3],
                                    split(name, "_")[5],
                                    split(split(name, "_")[7], ".")[1] *
                                    "." *
                                    split(split(name, "_")[7], ".")[2]))
                            for name in sender_is_1_csv_names]
    ##
    return weight_and_locations
end
