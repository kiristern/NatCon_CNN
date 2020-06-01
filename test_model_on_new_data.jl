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
#@time include("train_model.jl")
@time include("preprocess_idx.jl")


#have a look of trained model performance
begin
    print("##################")
    print("## Plotting...  ##")
    print("##################")
end
p1 = heatmap(validation_set[1][2][:,:,1,1], title="predicted") #connectivity map
p2 = heatmap(model(validation_set[1][1])[:,:,1,1], title="observed") #resistance and origin layer map
p3 = scatter(validation_set[1][2][:,:,1,1], model(validation_set[1][1])[:,:,1,2], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed", yaxis="predicted")
plot(p1,p2, p3)
savefig("figures/$(Stride)x$(Stride)_$(run)sec_$(best_acc*100)%.png")

validation_set

#=
Test model on:

nine_nine: set of R&O layers and C layer

check accuracy with:
validate_connect27x27: vector of m elements of dims 27x27

=#

model_on_9x9 = []
for i in 1:length(nine_nine)
    m = model(nine_nine[i][1])
    push!(model_on_9x9, m)
end
model_on_9x9

mod = []
for t in model_on_9x9
  tmp2 = [t[:,:,1,i] for i in 1:batch_size]
  push!(mod, tmp2)
end
mod
mod = reduce(vcat, mod)
stitched = [reduce(hcat, p) for p in Iterators.partition(mod, 3)]
stitchedmap = [reduce(vcat, p) for p in Iterators.partition(stitched[1:149], 3)]
plot(heatmap(stitchedmap[1]), heatmap(validate_connect27x27[1]))

scatterplotmaps = scatter(stitchedmap[1], validate_connect27x27[1], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed", yaxis="predicted")
plot(heatmap(stitchedmap[1]), heatmap(validate_connect27x27[1]), scatterplotmaps)
