#=

Create and train a model

=#

cd(@__DIR__)
@time include("libraries.jl")
@time include("functions.jl")
@time include("preprocess.jl")
@time include("validation_dataset.jl")
@time include("minibatch.jl")
@time include("model.jl")
@time @load "BSON/overwrite_9x9.bson" params #upload last saved model
Flux.loadparams!(model, params) #new model will now be identical to the one saved params for
# @time include("train_model.jl")
@time include("preprocess_idx.jl")


#have a look of trained model performance
begin
    print("##################")
    print("## Plotting...  ##")
    print("##################")
end
p1 = heatmap(validation_set[1][2][:,:,1,1], title="predicted") #connectivity map
p2 = heatmap(model(validation_set[1][1])[:,:,1,1], title="observed") #resistance and origin layer map
p3 = scatter(validation_set[1][2][:,:,1,1], model(validation_set[1][1])[:,:,1,1], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
plot(p1,p2,p3)
# savefig("figures/$(Stride)x$(Stride)_$(run)sec_$(best_acc*100)%.png")

validation_set

#=
Test model on:
    nine_nine: set of R&O layers and C layer

Stitch together individual 9x9 maps into 27x27 maps

check accuracy with:
    validate_connect27x27: vector of m elements of dims 27x27
=#

#run trained model on new minibatched data
model_on_9x9 = trained_model(nine_nine)

#stitch together 27x27 maps
stitchedmap = stitch4d(model_on_9x9)

#plot
scatterplotmaps = scatter(stitchedmap[56], validate_connect27x27[56], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
plot(heatmap(stitchedmap[56]), heatmap(validate_connect27x27[56]), scatterplotmaps)
