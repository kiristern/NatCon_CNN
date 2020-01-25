#https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl
#https://spcman.github.io/getting-to-know-julia/deep-learning/vision/flux-cnn-zoo/

using Flux
using StatsBase
using Random
using CSV
using BSON
using Printf

# include("lib.jl")
# input = readasc("input/resistance.asc"; nd="NODATA")
# origin = readasc("input/origin.asc"; nd="NODATA")
# output = readasc("output/connectivity.asc")

input = CSV.read("data/resistance.csv")
output = CSV.read("data/connectivity.csv", delim="\t")
input = convert(Matrix, input)
output = convert(Matrix, output)

#For simplicity and testing, start with presence/absence
#change to binary
I = zeros(Int64, size(input))
for j in 1:length(input)
    if isnan(input[j])
        I[j] = 0
    else
         I[j] = 1
    end
end
I

out = zeros(Int64, size(input))
for j in 1:length(input)
    if isnan(input[j])
        out[j] = 0
    else
         out[j] = 1
    end
end
out

#Take 20 000 unique and random cartesian indices (constructing a CartesianIndices from an array makes a range of its indices--ie. points) from the output
Y = unique(rand(CartesianIndices(out), 20_000))
filter!(y -> !isnan(I[y]), Y) #if input is NaN, remove
filter!(y -> 23 <= Tuple(y)[1] <= (size(I, 1)-24) , Y) #filter first element of the tuple between: 23 and size of input-24
filter!(y -> 23 <= Tuple(y)[2] <= (size(I, 2)-24) , Y) #filter second element of the tuple between: 23 and size of input-24

#create a range around (a,b) points of input at position Y
X = Array{Float32,2}[]
for y in Y
    a, b = Tuple(y) #tuple will not change
    c = reshape(I[a-14:a+13,b-14:b+13], (28, 28)) #c is a Y-element 28x28 array, where tuples (a,b) of input at position Y, range between a/b-14:a/b+13
    #not (28,28,1,1) because make_my_minibatch adds extra dimensions ??
    push!(X, c)
end
X

#get list of possible outputs
labels = [out[y] for y in Y]

# Bundle images together with labels and group into minibatchess
function make_my_minibatch(Xx, Yy, my_idxs)
    Xx_batch = Array{Float32}(undef, size(Xx[1])..., 1, length(my_idxs))
    for i in 1:length(my_idxs)
        Xx_batch[:, :, :, i] = Float32.(Xx[my_idxs[i]])
    end
    Yy_batch = Flux.onehotbatch(Yy[my_idxs], 0:1) #one-hot possible outputs (numbers from 0:1)
    return (Xx_batch, Yy_batch)
end
batch_size = 128
my_mb_idxs = Base.Iterators.partition(1:length(X), batch_size)
my_train_set = [make_my_minibatch(X, labels, i) for i in my_mb_idxs]

#look how training and test data has been arranged
typeof(my_train_set) #tuple of x training data (4D Float32 array) and y labels (Flux.OneHotVector)

#look at the size of the first training batch
size(my_train_set[1][1]) #Width=28 x Height=28 x Channel=1 x NumberBatches=128

#look at size of first batch of y labels
size(my_train_set[1][2]) #each OneHotVector encodes the labelled digit (ie. 0-1)

#look at the first OneHotVector in the first batch
my_train_set[1][2][:,1]

# Prepare test set as one giant minibatch:
my_test_set = make_my_minibatch(X, labels, 1:length(X))


# Define our model.  We will use a simple convolutional architecture with three iterations of Conv -> ReLU -> MaxPool, followed by a final Dense layer that feeds into a softmax probability output.
my_model = Chain(
    # First convolution layer, operating upon a 28x28 image
    Conv((3, 3), 1=>16, pad=(1,1), relu),
    MaxPool((2,2)),
    #3x3 conv filter size that will slide over image detecting new features
    #input size is 1 (recal one batch is of size 28x28x1x128)
    #output size is 16 = create 16 new channels for every training digit in the branch
    #pad=(1,1): pads a single layer of zeros around the images meaning that the dimensions of the conv output can remain at 28x28
    #relu: activation function
    #MaxPool=second layer. (2,2) is the window size that slides over x reducing it to half the size whilst retaining the most important feature info for learning
    ##output from this layer can be viewed with: model[1:2](train_set[1][1]) with output dimensions 14x14x16x128

    # Layer 3: Second convolution, operating upon a 14x14 image on output from layer 2
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),
    #input is 16 (from layer 2)
    #output size of the layer will be 32
    #output from this layer can be viewed with model[1:3](train_set[1][1]) and has output dimensions 14x14x32x128
    #MaxPool reduces the dims in half again
    #output can be viewed with model[1:4](train_set[1][1]) and has dims 7x7x32x128

    # Fifth convolution, operating upon a 7x7 image
    Conv((3, 3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),
    #final output from layer 6 is 3x3x32x128

    # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N) which is where we get the 288 in the `Dense` layer below:
    x -> reshape(x, :, size(x, 4)),
    #reshape layer flattens the data from 4D to 2D suitable for the dense layer and training
    #output can be viewed with model[1:7](train_set[1][1]) and has dims 288x128 (288 comes from multiplying output of layer 6: 3x3x32)
    Dense(288, 10),
    #Dense(288, 10) is the final training layer and takes the input 288 and outputs a size of 10x128 (10 for 10 digits of 0-9...re: model zoo MNIST tutorial)
    Dense(10, 2, σ), #adjust for our data

    # Finally, softmax to get nice probabilities
    softmax, #outputs probabilities between 0 and 1 of which digit the model has predicted
)



# Make sure our model is nicely precompiled before starting our training loop
my_model(my_train_set[1][1])

 #`loss()` calculates the mean squared error loss between our prediction `y_hat`
my_loss(x, y) = Flux.mse(my_model(x), y)


my_accuracy(x, y) = mean(Flux.onecold(my_model(x)) .== Flux.onecold(y))


# Train our model with the given training set using the ADAM optimizer and printing out performance against the test set as we go.
opt = ADAM(0.001) #Learning Rate (η) 0.001


@info("Beginning training loop...")
best_acc = 0.0
last_improvement = 0
for epoch_idx in 1:100
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(my_loss, params(my_model), my_train_set, opt)

    # Calculate accuracy:
    acc = my_accuracy(my_test_set...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))

    # If our accuracy is good enough, quit out.
    if acc >= 0.95
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= best_acc
        @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
        BSON.@save joinpath(dirname(@__FILE__), "mnist_conv.bson") my_model epoch_idx acc
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
