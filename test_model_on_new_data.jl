

cd(@__DIR__)
@time include("libraries.jl")
@time include("functions.jl") #desired object found in line 23 of StitchLargeMap.jl script
@time include("preprocess.jl")
@time include("validation_dataset.jl")
@time include("minibatch.jl")
@time include("model.jl")
@time @load "BSON/multispmod10_sampleonlywheredata.bson" params #upload last saved model
Flux.loadparams!(model, params) #new model will now be identical to the one saved params for
@time include("StitchLargerMap.jl")


#have a look of trained model performance
begin
    print("##################")
    print("## Plotting...  ##")
    print("##################")
end

connect_true = validation_set[1][2][:,:,1,4]
connect_model = model(validation_set[1][1])[:,:,1,4]

p1 = heatmap(connect_true, title="True Connectivity") #connectivity map
p2 = heatmap(connect_model, title="Predicted Connectivity (model)") #resistance and origin layer map
p3 = scatter(connect_true, connect_model, leg=false, c=:black, xlim=(0,1), ylim=(0,1), yaxis="Predicted (model)", xaxis="True")
plot(p1,p2,p3)
# savefig("figures/2sp_orig_rl_600samples_$(run)sec_$(best_acc*100)%.png")

#save as csv files
# using DelimitedFiles
#
# convert(Matrix{Float32}, connect_true) |> f -> writedlm("true_connect9x9.csv", f)
# convert(Matrix{Float32}, connect_model) |> f -> writedlm("model_connect9x9.csv", f)

#=
Test model on:
    nine_nine: set of R&O layers and C layer

Stitch together individual 9x9 maps into 27x27 maps

check accuracy with:
    validate_connect27x27: vector of m elements of dims 27x27
=#

#run trained model on new minibatched data (from )
model_on_9x9 = trained_model(nine_nine)

#stitch together 27x27 maps
stitchedmap = stitch4d(model_on_9x9)

#plot
scatterplotmaps = scatter(stitchedmap[70], validation_connectivity_map[70], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
plot(heatmap(stitchedmap[70]), heatmap(validation_connectivity_map[70]), scatterplotmaps)
# savefig("figures/2sp_orig_rl_300samples_45x45[70].png")










###### Compare connectivity layers between species #######
valid_connect_carcajou == valid_connect_cougar
valid_connect_carcajou == valid_connect_ours
valid_connect_cougar == valid_connect_ours

all(isapprox.(valid_connect_carcajou, valid_connect_cougar))
all(isapprox.(valid_connect_carcajou, valid_connect_ours))
all(isapprox.(valid_connect_cougar, valid_connect_ours))
