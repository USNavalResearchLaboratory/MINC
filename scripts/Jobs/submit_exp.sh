#!/bin/bash
## GPU 2
tmux kill-session -t gpu_2
tmux new-session -d -s gpu_2 'julia GPUs/gpu_2.jl'
## GPU 3
tmux kill-session -t gpu_3
tmux new-session -d -s gpu_3 'julia GPUs/gpu_3.jl'
## GPU 4
tmux kill-session -t gpu_4
tmux new-session -d -s gpu_4 'julia GPUs/gpu_4.jl'
## GPU 5
tmux kill-session -t gpu_5
tmux new-session -d -s gpu_5 'julia GPUs/gpu_5.jl'
## GPU 6
tmux kill-session -t gpu_6
tmux new-session -d -s gpu_6 'julia GPUs/gpu_6.jl'
## GPU 7
tmux kill-session -t gpu_7
tmux new-session -d -s gpu_7 'julia GPUs/gpu_7.jl'
