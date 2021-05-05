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


@time include("00libraries.jl")
# cd(@__DIR__)

#Read in the CSV file and convert them to arrays.
Resistance = readasc("data/maps_for_Kiri/Resistance_zone_beta_OursNoir.asc"; nd="NODATA")
Origin = readasc("data/input/origin.asc"; nd="NODATA")
Connectivity = readasc("data/maps_for_Kiri/Current_OursNoir.asc")

Stride = 9
number_of_samples = 150

@time include("functions.jl")

begin
  nan_to_0(Resistance)
  nan_to_0(Origin)
  nan_to_0(Connectivity)
end

#create Training and Testing datasets
# Extract 150 random 9x9 resistance, origin, and connectivity layers
maps, connect, test_maps, test_connect = make_datasets(Resistance, Origin, Connectivity)

#view points used for sampling
visual_samp_pts(get_train_samp1, get_train_samp2)

#script returns:
maps
connect
test_maps
test_connect
