#=

Create and train a model

=#

# cd(@__DIR__)
include("libraries.jl")
include("functions.jl")
include("Models/species.jl")
include("Models/multi_sp.jl")
# include("preprocess.jl")
# include("validation_dataset.jl")
# include("minibatch.jl")
include("model.jl")

# Load model and datasets onto GPU, if enabled
train_set = gpu.(train_set_multisp)
validation_set = gpu.(validation_set_multisp)
model = gpu(model)

# Make sure our model is nicely precompiled before starting our training loop
# model(train_set[1][1])
# model(train_set[1][1])[:, :, 1, 32] #see last output


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
    Flux.train!(loss, params(model), train_set, opt)

    #Terminate on NaN
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
        @info(" -> New best accuracy! Saving model out to BSON")
        BSON.@save joinpath(dirname(@__FILE__), "BSON/fox10_sampleonlywheredata.bson") #= TODO: make sure to change file name when training new model! =# params=cpu.(params(model)) epoch_idx acc
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

#have a look of trained model performance
begin
    print("##################")
    print("## Plotting...  ##")
    print("##################")
end
p1 = heatmap(validation_set[1][2][:,:,1,22], title="predicted") #connectivity map
p2 = heatmap(model(validation_set[1][1])[:,:,1,22], title="observed") #resistance and origin layer map
p3 = scatter(validation_set[1][2][:,:,1,22], model(validation_set[1][1])[:,:,1,22], leg=false, c=:black, xlim=(0,1), ylim=(0,1), yaxis="observed (model)", xaxis="predicted (true values)")
plot(p1,p2,p3)
savefig("figures/fox10_sampleonlywheredata_$(run)sec_$(best_acc*100)%_[$last_improvement].png")
