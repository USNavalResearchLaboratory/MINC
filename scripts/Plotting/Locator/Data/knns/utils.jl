## UTIL
function knn_dists(s::Int, sends)
    #
    send_s = sends[s]
    sends = stack(sends; dims=3)
    #
    tree = BruteTree(send_s)
    k = size(send_s, 2)
    point = dropdims(mean(sends; dims=(2, 3)); dims=(2, 3))
    idxs, dists = knn(tree, point, k, true)
    #
    return dists ./ norm(point)
end
