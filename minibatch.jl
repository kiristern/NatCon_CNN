include("validation_dataset.jl")

train_imgs
train_labels
valid_imgs
valid_labels


#bundle images together with labels and group into minibatches
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    #transform (10x10) to (10x10x1x#batch)
    Y_batch = Array{Float32}(undef, size(Y[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        Y_batch[:, :, :, i] = Float32.(Y[idxs[i]])
    end
    return (X_batch, Y_batch)
end
batch_size = 32
mb_idxs = Iterators.partition(1:length(train_imgs), batch_size)
train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]

#Check how data has been arranged
typeof(train_set) #tuple of 4D X_training data and 4D Y_labels
#Check dimensions: width x height x channels x #batches
size(train_set[1][1]) # 10x10x2x32
size(train_set[1][2]) # 10x10x1x32
