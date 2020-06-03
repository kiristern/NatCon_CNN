using Parameters: @with_kw

@with_kw mutable struct Args
    lr::Float64 = 3e-3
    epochs::Int = 20
    batch_size = 32
    savepath::String = "./"
end

@time include("libraries.jl")
@time include("functions.jl")
@time include("preprocess.jl")
@time include("validation_dataset.jl")

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



function get_processed_data(args)
    # Load labels and images from Flux.Data.MNIST
    train_labels = train_connect
    train_imgs = train_maps
    #subtract remainders to ensure all minibatches are the same length
    droplast = rem(length(train_maps), batch_size)
    mb_idxs = partition(1:length(train_imgs)-droplast, args.batch_size)
    train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]

    # Prepare test set as one giant minibatch:
    test_imgs = valid_maps
    test_labels = valid_connect
    droplast2 = rem(length(test_labels), batch_size)
    mb_idxs2 = partition(1:length(test_imgs)-droplast2, batch_size)
    test_set = [make_minibatch(test_imgs, test_labels, i) for i in mb_idxs2]

    return train_set, test_set
end


# Build model
function build_model(args; imgsize = (Stride,Stride,2))
    cnn_output_size = (imgsize[1],imgsize[2], 1, batch_size)

    return Chain(
    Conv((3, 3), imgsize[3]=>16, pad=(1,1), relu),
    MaxPool((2,2)),

    # Second convolution, operating upon a 4x4 image
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # # Third convolution, operating upon a 2x2 image
    # Conv((3, 3), 32=>32, pad=(1,1), relu),
    # MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
    flatten,
    Dense(Int(prod(size(model[1:4](train_set[1][1])))/batch_size), Stride*Stride),
    # Dense(prod(cnn_output_size), batch_size)
    x -> reshape(x, (cnn_output_size)))
end


# We augment `x` a little bit here, adding in random noise.
augment(x) = x .+ gpu(0.1f0*randn(eltype(x), size(x)))

# Returns a vector of all parameters used in model
paramvec(m) = vcat(map(p->reshape(p, :), params(m))...)

# Function to check if any element is NaN or not
anynan(x) = any(isnan.(x))

#Get accuracy per pixel (between true and predicted value)
accuracy(x, y) = 1 - mean(Flux.mse(model(x), y)) # (1 - mse) -> closer to 1 is better


function train(; kws...)
    args = Args(; kws...)

    @info("Loading data set")
    train_set, test_set = get_processed_data(args)

    # Define our model.  We will use a simple convolutional architecture with
    # three iterations of Conv -> ReLU -> MaxPool, followed by a final Dense layer.
    @info("Building model...")
    model = build_model(args)

    # Load model and datasets onto GPU, if enabled
    train_set = gpu.(train_set)
    test_set = gpu.(test_set)
    model = gpu(model)

    # Make sure our model is nicely precompiled before starting our training loop
    model(train_set[1][1])

    #calculate L2 loss between our prediction and "y_hat" (calculated from "model(x)") and the ground truth "y". Augment the data a bit by adding gaussian random noise to images to make it more robust
    function loss(x, y)
        x̂ = augment(x)
        ŷ = model(x̂)
        return sum((y .- ŷ).^2)./prod(size(x)) #divided by the actual value
    end

    # Train our model with the given training set using the ADAM optimizer and
    # printing out performance against the test set as we go.
    opt = ADAM(args.lr)

    @info("Beginning training loop...")
    best_acc = 0.0
    last_improvement = 0
    for epoch_idx in 1:args.epochs
        # Train for a single epoch
        Flux.train!(loss, params(model), train_set, opt)

        # Terminate on NaN
        if anynan(paramvec(model))
            @error "NaN params"
            break
        end

        # Calculate accuracy:
        acc = mean([accuracy(x, y) for (x, y) in test_set])

        @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
        # If our accuracy is good enough, quit out.
        if acc >= 0.999
            @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc >= best_acc
            @info(" -> New best accuracy! Saving model out to $(Stride)x$(Stride).bson")
            BSON.@save joinpath(args.savepath, "BSON/$(Stride)x$(Stride).bson") params=cpu.(params(model)) epoch_idx acc
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
end


train()

# Testing the model, from saved model
function test(; kws...)
    args = Args(; kws...)

    # Loading the test data
    _,test_set = get_processed_data(args)

    # Re-constructing the model with random initial weights
    model = build_model(args)

    # Loading the saved parameters
    BSON.@load joinpath(args.savepath, "BSON/9x9.bson") params

    # Loading parameters onto the model
    Flux.loadparams!(model, params)

    test_set = gpu.(test_set)
    model = gpu(model)
    @show accuracy(test_set...,model)
end

test()
