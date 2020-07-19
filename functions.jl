#=
Functions used in scripts
=#

#################
# Preprocessing #
#################

#Change NaN values to 0
function nan_to_0(s)
  for j in 1:length(s)
    if isnan(s[j])
      s[j] = 0
    end
  end
end



#####################
# Creating datasets #
#####################

#=
Create Training and Testing datasets
=#
#Extract 150 random 9x9 resistance, origin, and connectivity layers
number_of_samples = 150
bounds_connect = Connectivity[:,1:size(Connectivity,2)-Stride]
presence_data = findall(x->x > 0, bounds_connect)

Random.seed!(1234)
samp_pts = sample(presence_data, number_of_samples)

get_train_samp1 = []
get_train_samp2 = []
for i in 1:length(samp_pts)
  x = samp_pts[i][1]
  y = samp_pts[i][2]
  push!(get_train_samp1, y)
  push!(get_train_samp2, x)
end

Random.seed!(5678)
samp_pts2 = sample(presence_data, number_of_samples)

get_train_samp3 = []
get_train_samp4 = []
for i in 1:length(samp_pts2)
  x = samp_pts[i][1]
  y = samp_pts[i][2]
  push!(get_train_samp3, y)
  push!(get_train_samp4, x)
end

function make_datasets(Resistance, Origin, Connectivity)
  maps = []
  connect = []
  for i in get_train_samp2, j in get_train_samp1
    #taking groups of matrices of dimensions StridexStride
    x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
    x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
    x = cat(x_res, x_or, dims=3) #concatenate resistance and origin layers
    y = Connectivity[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
    # if minimum(y) > 0 #predict only when there is connectivity
      push!(maps, x)
      push!(connect, y)
    # end
  end

#create Testing dataset
  test_maps = []
  test_connect = []
  for i in get_train_samp4, j in get_train_samp3
    x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
    x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
    x = cat(x_res, x_or, dims=3)
    y = Connectivity[i:(i+Stride-1),j:(j+Stride-1)]
    #if minimum(y) > 0
      push!(test_maps, x)
      push!(test_connect, y)
    #end
  end
  return maps, connect, test_maps, test_connect
end



#=
Push samples from multiple species into maps_multisp, connect_multisp, test_multisp, test_maps_connect_multisp
=#
function samp_multi_sp(sp_res, sp_or, sp_con)
  push!(maps_multisp, make_datasets(sp_res, sp_or, sp_con)[1])
  push!(connect_multisp, make_datasets(sp_res, sp_or, sp_con)[2])
  push!(test_multisp, make_datasets(sp_res, sp_or, sp_con)[3])
  push!(test_maps_connect_multisp, make_datasets(sp_res, sp_or, sp_con)[4])
end


#=
Create Validation dataset
=#
function partition_dataset(maps, connect, valid_ratio=0.1, Shuffle=true)
  """
  Create a validation set from the training set

  Args:
    maps: array representing the image set from which the partitioning is made.
    connect: the connect associated with the provided images.
    valid_ratio (optional): the portion of the data that will be used in the validation set. Default: 0.1.
    shuffle (optional): whether or not to shuffle the data. Default: True.

  Return:
  A tuple of 4 elements (train_maps, train_connect, valid_maps, valid_connect) where:
    train_maps: an array of images for the training set.
    train_connect: connect associated with the images in the training set.
    valid_maps: an array of images for the validation set.
    valid_connect: connect associated with the images in the validation set.
  """
  if Shuffle == true
    indices = shuffle(collect(1:size(maps,1)))
  else
    indices = collect(1:size(maps,1))
  end

  n_training = Int(round((1.0 - valid_ratio)*length(indices)))
  train_idx, valid_idx = indices[1:n_training], indices[n_training+1:end]

  train_maps, valid_maps = maps[train_idx], maps[valid_idx]
  train_connect, valid_connect = connect[train_idx], connect[valid_idx]
  return train_maps, train_connect, valid_maps, valid_connect
end



#############
# Minibatch #
#############

#create minibatches
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    #transform (9x9) to (9x9x1x#batch)
    Y_batch = Array{Float32}(undef, size(Y[1])...,1, length(idxs))
    for i in 1:length(idxs)
        Y_batch[:, :, :, i] = Float32.(Y[idxs[i]])
    end
    return (X_batch, Y_batch)
end


#=
Create Train_set and validation_set
=#
function make_sets(train_maps, train_connect, valid_maps, valid_connect)
  #subtract remainders to ensure all minibatches are the same length
  droplast = rem(length(train_maps), batch_size)
  mb_idxs = Iterators.partition(1:length(train_maps)-droplast, batch_size)
  #train set in the form of batches
  train_set = [make_minibatch(train_maps, train_connect, i) for i in mb_idxs]

  droplast2 = rem(length(valid_maps), batch_size)
  mb_idxs2 = Iterators.partition(1:length(valid_maps)-droplast2, batch_size)
  #prepare validation set as one giant minibatch
  validation_set = [make_minibatch(valid_maps, valid_connect, i) for i in mb_idxs2]
  return train_set, validation_set
end







#########
# Model #
#########

# Augment `x`(input) a little bit here, adding in random noise.
augment(x) = x .+ gpu(0.1f0*randn(eltype(x), size(x)))
#returns a vector of all parameters used in model
paramvec(model) = vcat(map(p->reshape(p, :), params(model))...)
#check if any element is NaN or not
anynan(x) = any(isnan.(x))

#calculate L2 loss between our prediction and "y_hat" (calculated from "model(x)") and the ground truth "y". Augment the data a bit by adding gaussian random noise to images to make it more robust
function loss(x, y)
    x̂ = augment(x)
    ŷ = model(x̂)
    return sum((y .- ŷ).^2)./prod(size(x)) #divided by the actual value
end

#Get accuracy per pixel (between true and predicted value)
accuracy(x, y) = 1 - mean(Flux.mse(model(x), y)) # (1 - mse) -> closer to 1 is better





#################################
# Run trained model on new data #
#################################

#run trained model on new data
function trained_model(data9x9)
  model_on_data = [model(data9x9[i][1]) for i in 1:length(data9x9)]
  return model_on_data
end

#function to stitch 2D
function stitch2d(map)
  truemap = [reduce(hcat, p) for p in Iterators.partition(map, desired)]
  truemap = [reduce(vcat, p) for p in Iterators.partition(truemap, desired)]
  return truemap
end

#function to stitch together 3 (9x9) x 3 (9x9) to create one 27x27
function stitch4d(model_on_9x9)
  #reduce 4D to 2D
  mod = []
  for t in model_on_9x9
    tmp2 = [t[:,:,1,i] for i in 1:batch_size]
    push!(mod, tmp2)
  end
  #reduce to one vector of arrays
  mod = reduce(vcat, mod)
  #hcat groups of three
  stitched = [reduce(hcat, p) for p in Iterators.partition(mod, desired)]
  #vcat the stitched hcats
  stitchedmap = [reduce(vcat, p) for p in Iterators.partition(stitched[1:length(stitched)-1], desired)]
  return stitchedmap[1:length(stitchedmap)-1]
end
