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
p1 = heatmap(validation_set[1][2][:,:,1,2], title="predicted")
p2 = heatmap(model(validation_set[1][1])[:,:,1,2], title="observed")
p3 = scatter(validation_set[1][2][:,:,1,2], model(validation_set[1][1])[:,:,1,2], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed", yaxis="predicted")
plot(p1,p2, p3)
savefig("figures/$(Stride)x$(Stride)_$(run)sec_$(best_acc*100)%.png")

validation_set
#=
Test model on:

maps9x9: vector of n elements of dims 9x9x2
connect9x9: vector of n elements of dims 9x9

check accuracy with:
validation_maps: vector of m elements of dims 27x27x2
validation_connect: vector of m elements of dims 27x27

=#



maps9x9
connect9x9
validation_maps
validation_connect



#Stitching together 9 9x9 maps and compare to 27x27 map
v2 = validation_set2

model(v2[1][1])[:,:,1,2]
model(v2[1][1])

heatmap(v2[1][2][:,:,1,1])
heatmap(v2[1][2][:,:,1,2])
heatmap(v2[1][2][:,:,1,3])



v1 = validation_connect
heatmap(v1[1])


#have a look
begin
    print("##################")
    print("## Plotting...  ##")
    print("##################")
end
p1 = heatmap(validation_set2[1][2][:,:,1,2], title="predicted")
p2 = heatmap(model(validation_set2[1][1])[:,:,1,2], title="observed")
p3 = scatter(validation_set2[1][2][:,:,1,2], model(validation_set2[1][1])[:,:,1,2], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed", yaxis="predicted")
plot(p1,p2, p3)
savefig("figures/r2_$(Stride)x$(Stride)_$(run)sec_$(best_acc*100)%.png")
