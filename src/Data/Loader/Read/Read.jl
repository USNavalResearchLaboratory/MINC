## LOAD IN
include("signals.jl")
include("locations.jl")
##
function read_data(set::String)
    ##
    sender_folder_paths = readdir(projectdir("data/exp_raw/" * set); join=true)
    ## 
    signals, times = read_signals_times(sender_folder_paths) #[r, s, b](t)
    weight_and_locations = read_weight_and_locations(sender_folder_paths) #[w, x, y][b]
    ##
    loaded_indices = map(wal -> norm(wal) != 0, weight_and_locations)
    baseline_indices = map(wal -> norm(wal) == 0, weight_and_locations)
    # Baselines are (0, 0, 0)
    weight_and_locations_loaded = weight_and_locations[loaded_indices]
    ##
    locations = map(wal -> collect(wal[2:3]), weight_and_locations_loaded) #[i][b]
    ##
    signals_damaged = signals[:, :, loaded_indices]
    signals_baseline = signals[:, :, baseline_indices]
    ##
    return signals_damaged, signals_baseline, locations, times
end
##
