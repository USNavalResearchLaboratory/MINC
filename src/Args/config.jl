## GLOBAL CONFIG
Base.@kwdef mutable struct Config
    ## Model
    model_type::String = "A"
    ## Data set
    set::String = "AT3"
    grid_length::Int = 51
    p::Int = 2 # controls image rendering distance metric
    σ::Float32 = 1.5f0 # controls image rendering spread
    ## Dataloader
    ratio::Float32 = 0.80f0          # split data into train/test at ratio
    batch_size::Int = 32      # batch size
    partial::Bool = false  # controls discarding last minibatch if no of size batch_size    
    class_ratio::Float32 = 1.0f0
    ## Exp data processing
    ω_min::Real = 180 # kHz
    ω_max::Real = 419 # kHz
    """
    When discretized
        180 kHz -> 73
        419 kHz -> 169
    """
    t_start::Real = 0.0725 # ms
    t_end::Real = 0.40 # ms
    """
    In ms - 
        Begin first S0: 0.0725
        Begin first A0: 0.16
        End first A0: 0.24
        End signal: 0.4
    """
    σ_fknn::Real = 7 # controls discarding bad datapoints by send spectral density
    σ_rknn::Real = 1.65 # controls discarding bad datapoints by send waveform
    σ_maxabs::Real = 0.43 # controls discarding by amplitude (post baseline subtraction)
    ## Data preparation
    subtract_baseline::Bool = true # duplicate over baselines (no data leakage)
    zero_diagonals::Bool = true # discard sender noise
    ## CUDA
    dev = gpu_device()
    ## Logging
    infotime::Int = 25 #10      # report every `infotime` epochs, set to 0 for no checkpoints
    savepath::String = "runs/run/"   # results path
    ## Training (OneCycle + Optimiser + ClipGrad)
    Optimiser = Adam
    loss = md2e_iou_exc
    epochs::Int = 1_000 #500
    η_start::Float32 = 1.0f-5
    η::Float32 = 2.5f-3       # peak rate
    η_end::Float32 = 1.0f-3
    percent_start::Float32 = 0.20
    λ::Float32 = 1.0f-6              # weight decay
    δ::Float32 = 1.0f0              # ClipGrad
    ## Random number generator seeding
    PRNG = Xoshiro
    seed::Int = rand(1:(2^32))
    ##
    ordG::Int = 8 # Dihedral Group is of order 8.
    ## These fields are mutated! during training setup
    L_max::Int = 430 # max true x (or y) value for raw data origin
    L_min::Int = 180 # min true x (or y) value for raw data origin
    #
    ## Model A
    #
    A_activation::Function = swish
    A_connect_activation::Function = tanh_fast
    A_p::Float32 = 0.05
    # Equivariance
    A_L::Int = 1
    # Channels
    A_chs::Int = 16 #8
    #
    ## Model Z
    #
    Z_activation::Function = swish
    Z_connect_activation::Function = tanh_fast
    Z_p::Float32 = 0.05
    # Channels
    Z_chs::Int = 41 #19
    # Receiver/Sender index size
    r = 4
    s = 4
end
