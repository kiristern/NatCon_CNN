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

#read in datafiles
connectivity_carcajou = readasc("data/maps_for_Kiri/Current_Carcajou.asc")
connectivity_cougar = readasc("data/maps_for_Kiri/Current_Carcajou.asc")
connectivity_ours = readasc("data/maps_for_Kiri/Current_OursNoir.asc")

resistance_carcajou = readasc("data/maps_for_Kiri/Resistance_zone_beta_Carcajou.asc"; nd="NODATA")
resistance_cougar = readasc("data/maps_for_Kiri/Resistance_zone_beta_Cougar.asc"; nd="NODATA")
resistance_ours = readasc("data/maps_for_Kiri/Resistance_zone_beta_OursNoir.asc"; nd="NODATA")
resistance_coyote = readasc("data/maps_for_Kiri/Resistance_zone_beta_Coyote.asc"; nd="NODATA")

Origin = readasc("data/input/origin.asc"; nd="NODATA")

#convert NaN to zero
begin
  nan_to_0(connectivity_carcajou)
  nan_to_0(connectivity_cougar)
  nan_to_0(connectivity_ours)
  nan_to_0(resistance_carcajou)
  nan_to_0(resistance_cougar)
  nan_to_0(resistance_ours)
  nan_to_0(resistance_coyote)
  nan_to_0(Origin)
end

Stride = 9

maps_carcajou, connect_carcajou = training_dataset(resistance_carcajou, Origin, connectivity_carcajou)
test_maps_carcajou, test_connect_carcajou = testing_dataset(resistance_carcajou, Origin, connectivity_carcajou)

maps_cougar, connect_cougar = training_dataset(resistance_cougar, Origin, connectivity_cougar)
test_maps_cougar, test_connect_cougar = testing_dataset(resistance_cougar, Origin, connectivity_cougar)

maps_ours, connect_ours = training_dataset(resistance_cougar, Origin, connectivity_ours)
test_maps_ours, test_connect_ours = testing_dataset(resistance_cougar, Origin, connectivity_ours)
