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

#assign dimensions 28x28x1x1 (width x height x chanel (ex: greyscale=1, RGB=3) x number/batch)
#create a range around (a,b) points of input at position Y
X = Array{Float32,4}[]
for y in Y
    a, b = Tuple(y) #tuple will not change
    c = reshape(I[a-14:a+13,b-14:b+13], (28, 28, 1, 1)) #tuples range between a/b-14:a/b+13
    push!(X, c)
end
X


labels = [out[y] for y in Y]
#one-hot to transform output/labels into categories (either 0 or 1) with true/false
labels = Flux.onehotbatch(labels, 0:1)

# Partition into sets of 128 samples each to train the model
training_set = [(cat(float.(X[i])..., dims = 4), labels[:,i])
    for i in Base.Iterators.partition(1:length(X), 128)]

# Prepare test set (first 128 images)
tX = cat(X[1:128]..., dims = 4)
tY = Flux.onehotbatch(labels[1:128], 0:1)
testing_set = (tX, tY, 1:length(tX)) #error in output formatting here!! Training loop at point "accuracy(testing_set...)" doesn't run

size(training_set[1][1])
size(training_set[1][2])
training_set[1][2][:,1]


# Define our model.  We will use a simple convolutional architecture with
# three iterations of Conv -> ReLU -> MaxPool, followed by a final Dense
# layer that feeds into a softmax probability output.
my_model = Chain(
    Conv((3, 3), 1=>16, pad=(1,1), relu),
    MaxPool((2,2)),

    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    Conv((3, 3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10),

    Dense(10, 2, σ),
    #correct for our data (0-1)

    softmax,
)


# Make sure our model is nicely precompiled before starting our training loop
my_model(training_set[1][1])

# `loss()` calculates the mean squared error loss between our prediction `y_hat`
my_loss(x, y) = Flux.mse(my_model(x), y)


my_accuracy(x, y) = mean(onecold(my_model(x)) .== onecold(y))
# function accuracy(x, y)
#     a = onecold((model(x)))
#     b = onecold(y)  #### If this is not there, it beceomes a julia array
#     return mean(a .== b)
# end


# Train our model with the given training set using the ADAM optimizer and printing out performance against the test set as we go.
opt = ADAM(0.001) #Learning Rate (η) 0.001


# evalcb() = @show(loss(X[1], labels[1]))
# @Flux.epochs 5 Flux.train!(loss, params(model), training_set, opt, cb = Flux.throttle(evalcb, 5))

@info("Beginning training loop...")
best_acc = 0.0
last_improvement = 0
for epoch_idx in 1:100
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(my_loss, params(my_model), training_set, opt)

    # Calculate accuracy:
    acc = my_accuracy(testing_set...)
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
