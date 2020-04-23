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


########
# Second run
########
maps9x9
connect9x9

Random.seed!(1234)
train_maps2, train_connect2, valid_maps2, valid_connect2 = partition_dataset(maps9x9, connect9x9)
