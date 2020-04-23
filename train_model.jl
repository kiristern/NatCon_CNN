#=

Create and train a model

=#

cd(@__DIR__)
include("libraries.jl")
include("functions.jl")
include("preprocess.jl")
include("validation_dataset.jl")
include("minibatch.jl")


train_set
validation_set

m = Chain(
    Conv((3,3), 2=>16, pad=(1,1), relu),
    MaxPool((2,2)),
    Conv((3,3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2))
)

ls = m[1:4](train_set[1][1])
reshapeLayer = size(ls,1)*size(ls,2)*size(ls,3)

begin
    print("##########################")
    print("## Constructing model...##")
    print("##########################")
end
model = Chain(
    #Apply a Conv layer to a 2-channel input using a 2x2 window size, giving a 16-channel output. Output is activated by relu
    Conv((3,3), 2=>16, pad=(1,1), relu),
    MaxPool((2,2)),
    #2x2 window slides over x reducing it to half the size while retaining most important feature information for learning (takes highest/max value)
    Conv((3,3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    #flatten from 3D tensor to a 2D one, suitable for dense layer and training
    x -> reshape(x, (reshapeLayer, batch_size)),

     Dense(reshapeLayer, Stride*Stride),

    #reshape to match output dimensions
    x -> reshape(x, (Stride, Stride, 1, batch_size))
)

#View layer outputs
model[1](train_set[1][1]) #layer 1: 9x9x16x32
model[1:2](train_set[1][1]) #layer 2: 4x4x16x32
model[1:3](train_set[1][1]) #layer 3: 4x4x32x32
model[1:4](train_set[1][1]) #layer 4: 2x2x32x32
model[1:5](train_set[1][1]) #layer 5: 128x32
model[1:6](train_set[1][1]) #layer 6: 81x32
model[1:7](train_set[1][1]) #layer 7: 9x9x1x32

# Load model and datasets onto GPU, if enabled
train_set = gpu.(train_set)
validation_set = gpu.(validation_set)
model = gpu(model)

# Make sure our model is nicely precompiled before starting our training loop
model(train_set[1][1])
model(train_set[1][1])[:, :, 1, 32] #see last output


# Train our model with the given training set using the ADAM optimizer and printing out performance against the validation set as we go.
opt = ADAM(0.001)

# BSON.load("connectivity_relu.bson")
begin
    print("#################################")
    print("## Beginning training loop...  ##")
    print("#################################")
end
best_acc = 0.0
last_improvement = 0
run = @time @elapsed for epoch_idx in 1:5
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(loss, params(model), train_set, opt)

    if anynan(paramvec(model))
        @error "NaN params"
        break
    end

    # Calculate accuracy of model to validation set:
    acc = mean([accuracy(x, y) for (x, y) in validation_set]) #separating validation set tuple into the input and outputs & checking the accuracy between x and y; then getting mean
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))

    # If our accuracy is good enough, quit out.
    if acc >= 0.999
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= best_acc
        @info(" -> New best accuracy! Saving model out to connectivity.bson")
        BSON.@save joinpath(dirname(@__FILE__), "BSON/overwrite_$(Stride)x$(Stride).bson") params=cpu.(params(model)) epoch_idx acc
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

#have a look
begin
    print("##################")
    print("## Plotting...  ##")
    print("##################")
end
p1 = heatmap(validation_set[1][2][:,:,1,2], title="predicted")
p2 = heatmap(model(validation_set[1][1])[:,:,1,2], title="observed")
p3 = scatter(validation_set[1][2][:,:,1,2], model(validation_set[1][1])[:,:,1,2], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed", yaxis="predicted")
plot(p1,p2, p3)
savefig("figures/$(Stride)x$(Stride)_$(run)sec_$(best_acc*100)%.png")
