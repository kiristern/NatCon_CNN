#ex_input = [(Array{Float32}(undef, 3, 3, 2)), (Array{Float32}(undef, 3, 3, 2))]
#input = fill([ex_input], 5)
ex_output = Array{Float32}(undef, 3, 3)
output = fill(ex_output, 5)
ex_input = Array{Float32}(undef, 3, 3, 2)
input = fill(ex_input, 5)

function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = Array{Float32}(undef, size(Y[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        Y_batch[:, :, :, i] = Float32.(Y[idxs[i]])
    end
    return (X_batch, Y_batch)
end

batch_size = 32
mb_idxs = Iterators.partition(1:length(input), batch_size)
#train set in the form of batches
ex_train_set = [make_minibatch(input, output, i) for i in mb_idxs]

ex_validation_set = make_minibatch(input, output, 1:length(input))


model = Chain(
    Conv((3,3), 2=>16, pad=(1,1), relu),
    MaxPool((2,2)),

    Conv((3,3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    Conv((3,3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # #flatten from 3D tensor to a 2D one, suitable for dense layer and training
    x -> reshape(x, :, size(x, 4)),
     Dense(32, 10),
    # want final output dims 1x32
     Dense(10, 1, σ)

    #softmax to get nice probabilities
    #softmax,
)

function loss(x, y)
    x̂ = augment(x)
    ŷ = model(x̂)
    return crossentropy(ŷ, y) #TODO ensure input and output are same dimensions!!
end
