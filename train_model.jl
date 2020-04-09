include("dataloader.jl")
include("minibatch.jl")

using Flux, Statistics
using Flux: onecold, crossentropy
using Base.Iterators: repeated, partition
using Printf, BSON
using CUDAapi
if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end

train_set
validation_set

model = Chain(
    #Apply a Conv layer to a 2-channel input using a 2x2 window size, giving a 16-channel output. Output is activated by relu
    Conv((3,3), 2=>16, pad=(1,1), relu),
    #2x2 window slides over x reducing it to half the size while retaining most important feature information for learning (takes highest/max value)
    MaxPool((2,2)),

    Conv((3,3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    Conv((3,3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    #flattens the data from 4D to 2D suitable for dense layer and training
    x -> reshape(x, :, size(x, 4)),
    #takes output of previous layer (288) as input; and outputs a size of 10x32
    Dense(288, 10),
    #want final output dims 1x32
    Dense(10, 1, σ)

    #softmax to get nice probabilities
    softmax,
)

#View layer outputs
model[1](train_set[1][1]) #layer 1: 28x28x16x32
model[1:2](train_set[1][1]) #layer 2: 14x14x16x32
model[1:3](train_set[1][1]) #layer 3: 14x14x32x32
model[1:4](train_set[1][1]) #layer 4: 7x7x32x32
model[1:5](train_set[1][1]) #layer 5: 7x7x32x32
model[1:6](train_set[1][1]) #layer 6: 3x3x32x32
model[1:7](train_set[1][1]) #layer 7: 288x32 (288 = 3x3x32)
#softmax is to output probabilities of which label the model has predicted

# Load model and datasets onto GPU, if enabled
train_set = gpu.(train_set)
validation_set = gpu.(validation_set)
model = gpu(model)

# Make sure our model is nicely precompiled before starting our training loop
model(train_set[1][1])

# We augment `x` a little bit here, adding in random noise.
augment(x) = x .+ gpu(0.1f0*randn(eltype(x), size(x)))
paramvec(m) = vcat(map(p->reshape(p, :), params(m))...)
anynan(x) = any(isnan.(x))

# `loss()` calculates the crossentropy loss between our prediction `y_hat`
# (calculated from `model(x)`) and the ground truth `y`.  We augment the data
# a bit, adding gaussian random noise to our image to make it more robust.
function loss(x, y)
    x̂ = augment(x)
    ŷ = model(x̂)
    return crossentropy(ŷ, y)
end
accuracy(x, y) = mean(onecold(cpu(model(x))) .== onecold(cpu(y)))

# Train our model with the given training set using the ADAM optimizer and
# printing out performance against the test set as we go.
opt = ADAM(0.001)

@info("Beginning training loop...")
best_acc = 0.0
last_improvement = 0
@time @elapsed for epoch_idx in 1:100
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(loss, params(model), train_set, opt)

    if anynan(paramvec(model))
        @error "NaN params"
        break
    end

    # Calculate accuracy:
    acc = accuracy(validation_set...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))

    # If our accuracy is good enough, quit out.
    if acc >= 0.999
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= best_acc
        @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
        BSON.@save joinpath(dirname(@__FILE__), "mnist_conv.bson") params=cpu.(params(model)) epoch_idx acc
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





#evaluate callback
evalcb() = @show(loss(x_test, y_test))

epochs = 3

@info("Beginning training loop...")
Random.seed!(1234)
@time @elapsed for epoch in 1:epochs
    index = sample(1:length(X), epochs, replace=false)
    Flux.train!(loss, params(model), ncycle(train_loader, epochs), ADAM(0.001), cb = Flux.throttle(evalcb, 1))
end

#have a look
@info "plotting"
p1 = heatmap(Z[1], title="predicted")
p2 = heatmap(model(W[1]), title="observed")
p3 = scatter(Z[1], model(W[1]), leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed", yaxis="predicted")
plot(p1,p2, p3)
