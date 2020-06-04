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
connectivity_cougar = readasc("data/maps_for_Kiri/Current_cougar.asc")
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
batch_size = 32

#carcajou
maps_carcajou, connect_carcajou, test_maps_carcajou, test_connect_carcajou = make_datasets(resistance_carcajou, Origin, connectivity_carcajou)

train_maps_carcajou, train_connect_carcajou, valid_maps_carcajou, valid_connect_carcajou = partition_dataset(maps_carcajou, connect_carcajou)

train_set_carcajou, validation_set_carcajou = make_sets(train_maps_carcajou, train_connect_carcajou, valid_maps_carcajou, valid_connect_carcajou)




#cougar
maps_cougar, connect_cougar, test_maps_cougar, test_connect_cougar = make_datasets(resistance_cougar, Origin, connectivity_cougar)

train_maps_cougar, train_connect_cougar, valid_maps_cougar, valid_connect_cougar = partition_dataset(maps_cougar, connect_cougar)

train_set_cougar, validation_set_cougar = make_sets(train_maps_cougar, train_connect_cougar, valid_maps_cougar, valid_connect_cougar)



#ours noir
maps_ours, connect_ours, test_maps_ours, test_connect_ours = make_datasets(resistance_cougar, Origin, connectivity_ours)

train_maps_ours, train_connect_ours, valid_maps_ours, valid_connect_ours = partition_dataset(maps_ours, connect_ours)

train_set_ours, validation_set_ours = make_sets(train_maps_ours, train_connect_ours, valid_maps_ours, valid_connect_ours)
