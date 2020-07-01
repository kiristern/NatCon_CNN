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


# include("libraries.jl")
# include("functions.jl")
# cd(@__DIR__)


#create Training dataset
# Extract 150 random 9x9 resistance, origin, and connectivity layers
Random.seed!(1234)
Stride = 9


maps_multisp, connect_multisp, test_multisp, test_maps_connect_multisp = [], [], [], []

push!(maps_multisp, make_datasets(resistance_carcajou, Origin, connectivity_carcajou)[1])
push!(connect_multisp, make_datasets(resistance_carcajou, Origin, connectivity_carcajou)[2])
push!(test_multisp, make_datasets(resistance_carcajou, Origin, connectivity_carcajou)[3])
push!(test_maps_connect_multisp, make_datasets(resistance_carcajou, Origin, connectivity_carcajou)[4])

push!(maps_multisp, make_datasets(resistance_ours, Origin, connectivity_ours)[1])
push!(connect_multisp, make_datasets(resistance_ours, Origin, connectivity_ours)[2])
push!(test_multisp, make_datasets(resistance_ours, Origin, connectivity_ours)[3])
push!(test_maps_connect_multisp, make_datasets(resistance_ours, Origin, connectivity_ours)[4])

push!(maps_multisp, make_datasets(resistance_cougar, Origin, connectivity_cougar)[1])
push!(connect_multisp, make_datasets(resistance_cougar, Origin, connectivity_cougar)[2])
push!(test_multisp, make_datasets(resistance_cougar, Origin, connectivity_cougar)[3])
push!(test_maps_connect_multisp, make_datasets(resistance_cougar, Origin, connectivity_cougar)[4])

push!(maps_multisp, make_datasets(resistance_renard, Origin, connectivity_renard)[1])
push!(connect_multisp, make_datasets(resistance_renard, Origin, connectivity_renard)[2])
push!(test_multisp, make_datasets(resistance_renard, Origin, connectivity_renard)[3])
push!(test_maps_connect_multisp, make_datasets(resistance_renard, Origin, connectivity_renard)[4])

push!(maps_multisp, make_datasets(resistance_ratonlaveur, Origin, connectivity_ratonlaveur)[1])
push!(connect_multisp, make_datasets(resistance_ratonlaveur, Origin, connectivity_ratonlaveur)[2])
push!(test_multisp, make_datasets(resistance_ratonlaveur, Origin, connectivity_ratonlaveur)[3])
push!(test_maps_connect_multisp, make_datasets(resistance_ratonlaveur, Origin, connectivity_ratonlaveur)[4])

maps_multisp
connect_multisp
test_multisp
test_maps_connect_multisp

maps_multisp = vcat(maps_multisp...)
connect_multisp = vcat(connect_multisp...)


train_maps_multisp, train_connect_multisp, valid_maps_multisp, valid_connect_multisp = partition_dataset(maps_multisp, connect_multisp)

train_set_multisp, validation_set_multisp = make_sets(train_maps_multisp, train_connect_multisp, valid_maps_multisp, valid_connect_multisp)



#script returns:
maps_multisp
connect_multisp
test_maps_multisp
test_connect_multisp




train_maps_multisp, train_connect_multisp, valid_maps_multisp, valid_connect_multisp = partition_dataset(maps_multisp, connect_multisp)

train_set_multisp, validation_set_multisp = make_sets(train_maps_multisp, train_connect_multisp, valid_maps_multisp, valid_connect_multisp)

#TODO: run on train_model.jl script! :D
