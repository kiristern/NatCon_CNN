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

maps_2sp = []
connect_2sp = []
for i in rand(10:950, 75), j in rand(10:950, 75)
  #taking groups of matrices of dimensions StridexStride
  x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin layers
  y = Connectivity[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  if minimum(y) > 0 #predict only when there is connectivity
    push!(maps_2sp, x)
    push!(connect_2sp, y)
  end
end
for i in rand(10:950, 75), j in rand(10:950, 75)
  #taking groups of matrices of dimensions StridexStride
  x_res = resistance_ratonlaveur[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin layers
  y = connectivity_ratonlaveur[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  if minimum(y) > 0 #predict only when there is connectivity
    push!(maps_2sp, x)
    push!(connect_2sp, y)
  end
end

#create Testing dataset
Random.seed!(5678)

test_maps_2sp = []
test_connect_2sp = []
for i in rand(10:950, 75), j in rand(10:950, 75)
  #taking groups of matrices of dimensions StridexStride
  x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin vectors
  y = Connectivity[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  if minimum(y) > 0 #predict only when there is connectivity
    push!(test_maps_2sp, x)
    push!(test_connect_2sp, y)
  end
end
for i in rand(10:950, 75), j in rand(10:950, 75)
  #taking groups of matrices of dimensions StridexStride
  x_res = resistance_ratonlaveur[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin vectors
  y = connectivity_ratonlaveur[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  if minimum(y) > 0 #predict only when there is connectivity
    push!(test_maps_2sp, x)
    push!(test_connect_2sp, y)
  end
end

#script returns:
maps_2sp
connect_2sp
test_maps_2sp
test_connect_2sp




train_maps_2sp, train_connect_2sp, valid_maps_2sp, valid_connect_2sp = partition_dataset(maps_2sp, connect_2sp)

train_set_2sp, validation_set_2sp = make_sets(train_maps_2sp, train_connect_2sp, valid_maps_2sp, valid_connect_2sp)

#TODO: run on train_model.jl script! :D 
