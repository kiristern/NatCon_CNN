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
include("model.jl")
# cd(@__DIR__)

#read in datafiles
connectivity_carcajou = readasc("data/maps_for_Kiri/Current_Carcajou.asc")
connectivity_cougar = readasc("data/maps_for_Kiri/Current_cougar.asc")
connectivity_ours = readasc("data/maps_for_Kiri/Current_OursNoir.asc")
connectivity_renard = readasc("data/maps_for_Kiri/RR_cum_currmap.asc")
connectivity_ratonlaveur = readasc("data/maps_for_Kiri/RL_cum_currmap.asc")

resistance_carcajou = readasc("data/maps_for_Kiri/Resistance_zone_beta_Carcajou.asc"; nd="NODATA")
resistance_cougar = readasc("data/maps_for_Kiri/Resistance_zone_beta_Cougar.asc"; nd="NODATA")
resistance_ours = readasc("data/maps_for_Kiri/Resistance_zone_beta_OursNoir.asc"; nd="NODATA")
resistance_coyote = readasc("data/maps_for_Kiri/Resistance_zone_beta_Coyote.asc"; nd="NODATA")
resistance_renard = readasc("data/maps_for_Kiri/Resistance_zone_beta_RR.asc"; nd="NODATA")
resistance_ratonlaveur = readasc("data/maps_for_Kiri/Resistance_zone_beta_RL.asc"; nd="NODATA")

Origin = readasc("data/input/origin.asc"; nd="NODATA")

#convert NaN to zero
begin
  nan_to_0(connectivity_carcajou)
  nan_to_0(connectivity_cougar)
  nan_to_0(connectivity_ours)
  nan_to_0(connectivity_renard)
  nan_to_0(connectivity_ratonlaveur)
  nan_to_0(resistance_carcajou)
  nan_to_0(resistance_cougar)
  nan_to_0(resistance_ours)
  nan_to_0(resistance_coyote)
  nan_to_0(resistance_renard)
  nan_to_0(resistance_ratonlaveur)
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





#renard roux
maps_renard, connect_renard, test_maps_renard, test_connect_renard = make_datasets(resistance_renard, Origin, connectivity_renard)

train_maps_renard, train_connect_renard, valid_maps_renard, valid_connect_renard = partition_dataset(maps_renard, connect_renard)

train_set_renard, validation_set_renard = make_sets(train_maps_renard, train_connect_renard, valid_maps_renard, valid_connect_renard)







#raton laveur
maps_ratonlaveur, connect_ratonlaveur, test_maps_ratonlaveur, test_connect_ratonlaveur = make_datasets(resistance_ratonlaveur, Origin, connectivity_ratonlaveur)

train_maps_ratonlaveur, train_connect_ratonlaveur, valid_maps_ratonlaveur, valid_connect_ratonlaveur = partition_dataset(maps_ratonlaveur, connect_ratonlaveur)

train_set_ratonlaveur, validation_set_ratonlaveur = make_sets(train_maps_ratonlaveur, train_connect_ratonlaveur, valid_maps_ratonlaveur, valid_connect_ratonlaveur)
