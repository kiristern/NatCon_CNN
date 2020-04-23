#=
Create a custom validation set by sampling from the training dataset

Input:
  maps: nxnx2
  connect: nxn

Output:
  train_maps: nxnx2
  train_connect: nxn
  valid_maps: nxnx2
  valid_connect: nxn
=#

# include("functions.jl")
# include("preprocess.jl")


######
# First run
######
maps
connect

Random.seed!(1234)
train_maps, train_connect, valid_maps, valid_connect = partition_dataset(maps, connect)
