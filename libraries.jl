# import Pkg; Pkg.activate(".") #must uncomment if trying to train, otherwise if uploading saved bson files, do not run
using StatsBase
using Random
using Flux, Statistics
using Flux: onecold, crossentropy
using Base.Iterators: repeated, partition
using Printf, BSON
using BSON: @load, @save
using CUDAapi
using Plots

if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end

include("lib.jl")
