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

#read in datafiles
begin
  connectivity_carcajou = readasc("data/maps_for_Kiri/Current_Carcajou.asc")
  connectivity_cougar = readasc("data/maps_for_Kiri/Current_Cougar.asc")
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
end

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


#save as csv files
# using DelimitedFiles
#
# convert(Matrix{Float32}, connectivity_carcajou) |> f -> writedlm("connectivity_carcajou.csv", f)
# convert(Matrix{Float32}, connectivity_cougar) |> f -> writedlm("connectivity_cougar.csv", f)
# convert(Matrix{Float32}, connectivity_ours) |> f -> writedlm("connectivity_oursnoir.csv", f)
# convert(Matrix{Float32}, connectivity_renard) |> f -> writedlm("connectivity_renard.csv", f)
# convert(Matrix{Float32}, connectivity_ratonlaveur) |> f -> writedlm("connectivity_ratonlaveur.csv", f)
# convert(Matrix{Float32}, resistance_carcajou) |> f -> writedlm("resistance_carcajou.csv", f)
# convert(Matrix{Float32}, resistance_cougar) |> f -> writedlm("resistance_cougar.csv", f)
# convert(Matrix{Float32}, resistance_ours) |> f -> writedlm("resistance_oursnoir.csv", f)
# convert(Matrix{Float32}, resistance_coyote) |> f -> writedlm("resistance_coyote.csv", f)
# convert(Matrix{Float32}, resistance_renard) |> f -> writedlm("resistance_renard.csv", f)
# convert(Matrix{Float32}, resistance_ratonlaveur) |> f -> writedlm("resistance_ratonlaveur.csv", f)
# convert(Matrix{Float32}, Origin) |> f -> writedlm("Origin.csv", f)


#create Training dataset
# Extract 150 random 9x9 resistance, origin, and connectivity layers
Random.seed!(1234)
Stride = 9
batch_size=32
number_of_samples = 30

maps_multisp, connect_multisp, test_multisp, test_maps_connect_multisp = [], [], [], []

samp_multi_sp(resistance_carcajou, Origin, connectivity_carcajou)
samp_multi_sp(resistance_cougar, Origin, connectivity_cougar)
samp_multi_sp(resistance_ours, Origin, connectivity_ours)
samp_multi_sp(resistance_renard, Origin, connectivity_renard)
samp_multi_sp(resistance_ratonlaveur, Origin, connectivity_ratonlaveur)


maps_multisp
connect_multisp
test_multisp
test_maps_connect_multisp

maps_multisp = vcat(maps_multisp...)
connect_multisp = vcat(connect_multisp...)


#script returns:
# maps_multisp
# connect_multisp
# test_maps_multisp
# test_connect_multisp


train_maps_multisp, train_connect_multisp, valid_maps_multisp, valid_connect_multisp = partition_dataset(maps_multisp, connect_multisp)

train_set_multisp, validation_set_multisp = make_sets(train_maps_multisp, train_connect_multisp, valid_maps_multisp, valid_connect_multisp)

get_train_samp1
visual_samp_pts(get_train_samp1, get_train_samp2)


# include("model.jl")
#TODO: run on train_model.jl script! :D
