begin
    print("##########################")
    print("## loading libraries...##")
    print("##########################")
end

using StatsBase
using CSV
using Random
using Flux, Statistics
using Flux: onecold, crossentropy
using Base.Iterators: repeated, partition
using Printf, BSON
using BSON: @load, @save
using CUDAapi
using Plots

if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end

#=
Functions used in scripts
=#


begin
    print("##########################")
    print("## loading functions...##")
    print("##########################")
end



#Change NaN values to 0
function nan_to_0(s)
  for j in 1:length(s)
    if isnan(s[j])
      s[j] = 0
    end
  end
end



#=
Create Training and Testing datasets
=#
#Extract 150 random 9x9 resistance, origin, and connectivity layers
function make_datasets(Resistance, Origin, Connectivity)
  Random.seed!(1234)
  maps = []
  connect = []
  for i in rand(1:size(Origin,2)-Stride, 150), j in rand(1:size(Origin,2)-Stride, 150)
    #taking groups of matrices of dimensions StridexStride
    x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
    x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
    x = cat(x_res, x_or, dims=3) #concatenate resistance and origin layers
    y = Connectivity[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
    #if minimum(y) > 0 #predict only when there is connectivity
      push!(maps, x)
      push!(connect, y)
    #end
  end
#create Testing dataset
  Random.seed!(5678)
  test_maps = []
  test_connect = []
  for i in rand(1:size(Origin,2)-Stride, 150), j in rand(1:size(Origin,2)-Stride, 150)
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


begin
    print("##########################")
    print("## importing data...##")
    print("##########################")
end


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

begin
    print("##########################")
    print("## preprocessing data...##")
    print("##########################")
end


Stride = 9
batch_size = 32

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




maps_multisp = vcat(maps_multisp...)
connect_multisp = vcat(connect_multisp...)




train_maps_multisp, train_connect_multisp, valid_maps_multisp, valid_connect_multisp = partition_dataset(maps_multisp, connect_multisp)



train_set_multisp, validation_set_multisp = make_sets(train_maps_multisp, train_connect_multisp, valid_maps_multisp, valid_connect_multisp)


train_maps_multisp, train_connect_multisp, valid_maps_multisp, valid_connect_multisp = partition_dataset(maps_multisp, connect_multisp)

train_set_multisp, validation_set_multisp = make_sets(train_maps_multisp, train_connect_multisp, valid_maps_multisp, valid_connect_multisp)



begin
    print("##########################")
    print("## Constructing model...##")
    print("##########################")
end

m = Chain(
    Conv((3,3), 2=>16, pad=(1,1), relu),
    MaxPool((2,2)),
    Conv((3,3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2))
)

inputlayersize = Array{Float32}(undef, 9, 9, 2, 32)

model = Chain(
    #Apply a Conv layer to a 2-channel (R & O layer) input using a 2x2 window size, giving a 16-channel output. Output is activated by relu
    Conv((3,3), 2=>16, pad=(1,1), relu),
    MaxPool((2,2)),
    #2x2 window slides over x reducing it to half the size while retaining most important feature information for learning (takes highest/max value)
    Conv((3,3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    #flatten from 3D tensor to a 2D one, suitable for dense layer and training
    x -> reshape(x, (Int(prod(size(m[1:4](inputlayersize)))/batch_size), batch_size)),

     Dense(Int(prod(size(m[1:4](inputlayersize)))/batch_size), Stride*Stride),

    #reshape to match output dimensions
    x -> reshape(x, (Stride, Stride, 1, batch_size))
)

begin
    print("##########################")
    print("## load on GPU...##")
    print("##########################")
end

# Load model and datasets onto GPU, if enabled
train_set = gpu.(train_set_multisp)
validation_set = gpu.(validation_set_multisp)
model = gpu(model)

# Make sure our model is nicely precompiled before starting our training loop
# model(train_set[1][1])
# model(train_set[1][1])[:, :, 1, 32] #see last output
begin
    print("##########################")
    print("## define optimizer, best acc and last improvement...##")
    print("##########################")
end

# Train our model with the given training set using the ADAM optimizer and printing out performance against the validation set as we go.
opt = ADAM(0.001)
best_acc = 0.0
last_improvement = 0
begin
    print("#################################")
    print("## Beginning training loop...  ##")
    print("#################################")
end
run = @time @elapsed for epoch_idx in 1:200
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(loss, params(model), train_set_multisp, opt)

    #Terminate on NaN
    if anynan(paramvec(model))
        @error "NaN params"
        break
    end

    # Calculate accuracy of model to validation set:
    acc = mean([accuracy(x, y) for (x, y) in validation_set_multisp]) #separating validation set tuple into the input and outputs & checking the accuracy between x and y; then getting mean
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))

    # If our accuracy is good enough, quit out.
    if acc >= 0.999
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= best_acc
        @info(" -> New best accuracy! Saving model out to BSON")
        BSON.@save joinpath(dirname(@__FILE__), "BSON/allspecies.bson") #= TODO: make sure to change file name when training new model! =# params=cpu.(params(model)) epoch_idx acc
        best_acc = acc
        last_improvement = epoch_idx
    end

    # If we haven't seen improvement in 5 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
        opt.eta /= 10.0
        @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end

    if epoch_idx - last_improvement >= 10
        @warn(" -> We're calling this converged.")
        break
    end
end
