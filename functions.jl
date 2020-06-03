#=
Functions used in scripts
=#

#Change NaN values to 0
function nan_to_0(s)
  for j in 1:length(s)
    if isnan(s[j])
      s[j] = 0
    end
  end
end


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



#Create validation set
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

#run trained model on new data
function trained_model(data)
  model_on_data = [model(data[i][1]) for i in 1:length(data)]
  return model_on_data
end

#function to stitch 2D
function stitch2d(map)
  truemap = [reduce(hcat, p) for p in Iterators.partition(map, 3)]
  truemap = [reduce(vcat, p) for p in Iterators.partition(truemap, 3)]
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
  stitched = [reduce(hcat, p) for p in Iterators.partition(mod, 3)]
  #vcat the stitched hcats
  stitchedmap = [reduce(vcat, p) for p in Iterators.partition(stitched[1:length(stitched)-1], 3)]
  return stitchedmap[1:length(stitchedmap)-1]
end
