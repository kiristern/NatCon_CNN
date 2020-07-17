#=
This script loads the original CSV datafiles and returns m nxn random samples for training and testing sets

Input:
  resistance.csv
  origin.csv
  connectivity.csv

Output:
  maps: x_train
  connect: y_train
  test_maps: x_test
  test_connect: y_test
=#


include("libraries.jl")
include("functions.jl")
# cd(@__DIR__)

#Read in the data
Resistance = readasc("data/input/resistance.asc"; nd="NODATA")
Origin = readasc("data/input/origin.asc"; nd="NODATA")
Connectivity = readasc("data/output/connectivity.asc")

begin
  nan_to_0(Resistance)
  nan_to_0(Origin)
  nan_to_0(Connectivity)
end

#create Training dataset
# Extract 150 random 9x9 resistance, origin, and connectivity layers
Stride = 9
Random.seed!(1234)
get_train_samp = rand(Stride:size(Origin,2)-Stride, 150)
get_train_samp2 = rand(Stride:size(Origin,2)-Stride, 150)

maps = []
connect = []
for i in get_train_samp, j in get_train_samp2
  #taking groups of matrices of dimensions StridexStride
  x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin layers
  y = Connectivity[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  push!(maps, x)
  push!(connect, y)
end

#create Testing dataset
Random.seed!(5678)
get_train_samp3 = rand(Stride:size(Origin,2)-Stride, 150)
get_train_samp4 = rand(Stride:size(Origin,2)-Stride, 150)

test_maps = []
test_connect = []
for i in get_train_samp3, j in get_train_samp4
  #taking groups of matrices of dimensions StridexStride
  x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin vectors
  y = Connectivity[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  push!(test_maps, x)
  push!(test_connect, y)
end

#script returns:
maps
connect
test_maps
test_connect
