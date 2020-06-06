import Pkg; Pkg.activate(".")
using StatsBase
using CSV
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
