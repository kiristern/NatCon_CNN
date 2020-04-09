include("validation_dataset.jl")

train_maps
train_connect
valid_maps
valid_connect


#bundle images together with connect and group into minibatches
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    #transform (28x28) to (28x28x1x#batch)
    Y_batch = Array{Float32}(undef, size(Y[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        Y_batch[:, :, :, i] = Float32.(Y[idxs[i]])
    end
    return (X_batch, Y_batch)
end
# The CNN only "sees" 32 images at each training cycle:
batch_size = 32
mb_idxs = Iterators.partition(1:length(train_maps), batch_size)
#train set in the form of batches
train_set = [make_minibatch(train_maps, train_connect, i) for i in mb_idxs]
#train set in one-go: used to calculate accuracy with the train set
train_set_full = make_minibatch(train_maps, train_connect, 1:length(train_maps))


#Check how data has been arranged
typeof(train_set) #tuple of 4D X_training data and 4D Y_connect
#Check dimensions: width x height x channels x #batches
size(train_set[1][1]) # 10x10x2x32
size(train_set[1][2]) # 10x10x1x32

#prepare validation set as one giant minibatch
validation_set = make_minibatch(valid_maps, valid_connect, 1:length(valid_maps))
