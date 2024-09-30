##
function window_time_raw(t_start, t_end)
    t_start_idx = round(Int, t_start * (10_000 / 0.4))
    t_end_idx = round(Int, t_end * (10_000 / 0.4))
    return t_start_idx:t_end_idx
end
