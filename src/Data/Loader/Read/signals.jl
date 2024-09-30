## SIGNALS/TIMES
function read_signals_times(sender_folder_paths::Vector{String})
    """
    Create dataframes containing time/voltage signals

    :::warning

    C.R. labels the piezos in the provided data as follows:

    |2 4|
    |1 3|.

    However, J.A. would prefer that a counterclockwise ordering is used

    |2 1|
    |3 4|,

    as this is a common standard that is assumed in our construction
    of the defining representation of the square group.
    :::
    """
    ##
    sender_is_1_csv_paths = readdir(sender_folder_paths[4]; join=true)
    sender_is_2_csv_paths = readdir(sender_folder_paths[2]; join=true)
    sender_is_3_csv_paths = readdir(sender_folder_paths[1]; join=true)
    sender_is_4_csv_paths = readdir(sender_folder_paths[3]; join=true)
    ##
    sender_is_1_dfs = [CSV.read(csv_path, DataFrame; header=0, types=Float32)
                       for csv_path in sender_is_1_csv_paths]
    sender_is_2_dfs = [CSV.read(csv_path, DataFrame; header=0, types=Float32)
                       for csv_path in sender_is_2_csv_paths]
    sender_is_3_dfs = [CSV.read(csv_path, DataFrame; header=0, types=Float32)
                       for csv_path in sender_is_3_csv_paths]
    sender_is_4_dfs = [CSV.read(csv_path, DataFrame; header=0, types=Float32)
                       for csv_path in sender_is_4_csv_paths]
    ## Time arrays are all the same 
    sender_is_1_times = [df.Column1 for df in sender_is_1_dfs]
    sender_is_2_times = [df.Column1 for df in sender_is_2_dfs]
    sender_is_3_times = [df.Column1 for df in sender_is_3_dfs]
    sender_is_4_times = [df.Column1 for df in sender_is_4_dfs]
    times = sender_is_1_times[1]
    ## Get voltages from the DataFrame
    """
    :::warning

    C.R. labels the piezos in the provided data as follows:

    |2 4|
    |1 3|.

    However, J.A. would prefer that a counterclockwise ordering is used

    |2 1|
    |3 4|,

    as this is a common standard that is assumed in our construction
    of the defining representation of the square group.

    Column1 - Time
    Column2 - C.R. 1 = J.A. 3
    Column3 - C.R. 2 = J.A. 2
    Column4 - C.R. 3 = J.A. 4
    Column5 - C.R. 4 = J.A. 1
    :::
    """
    ##
    sender_is_1_voltages = [[df.Column5, df.Column3, df.Column2, df.Column4]
                            for df in sender_is_1_dfs]
    sender_is_2_voltages = [[df.Column5, df.Column3, df.Column2, df.Column4]
                            for df in sender_is_2_dfs]
    sender_is_3_voltages = [[df.Column5, df.Column3, df.Column2, df.Column4]
                            for df in sender_is_3_dfs]
    sender_is_4_voltages = [[df.Column5, df.Column3, df.Column2, df.Column4]
                            for df in sender_is_4_dfs]
    ##
    sender_is_1_voltages_reshaped = stack(sender_is_1_voltages)
    sender_is_2_voltages_reshaped = stack(sender_is_2_voltages)
    sender_is_3_voltages_reshaped = stack(sender_is_3_voltages)
    sender_is_4_voltages_reshaped = stack(sender_is_4_voltages)
    ## Index notation: r = Receiver, s = Sender, b = Batch 
    signals = [sender_is_1_voltages_reshaped, sender_is_2_voltages_reshaped,
        sender_is_3_voltages_reshaped, sender_is_4_voltages_reshaped] #[s][r, b](t)
    signals = stack(signals; dims=2) #[r, s, b](t)
    ##
    return signals, times
end
