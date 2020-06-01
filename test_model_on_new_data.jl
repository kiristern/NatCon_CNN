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

maps9x9: vector of n elements of dims 9x9x2
connect9x9: vector of n elements of dims 9x9

check accuracy with:
validation_maps: vector of m elements of dims 27x27x2
validation_connect: vector of m elements of dims 27x27

=#

model_on_9x9 = []
for i in 1:length(nine_nine)
    m = model(nine_nine[i][1])
    push!(model_on_9x9, m)
end
model_on_9x9

heatmap(model_on_9x9[1][:,:,1,3])

model_on_9x9[1]

things = []
n = Int(length(model_on_9x9)/9)
for i in 1:n
    thing = vcat((hcat(model_on_9x9[:,:,1,i], model_on_9x9[:,:,1,i+1], model_on_9x9[:,:,1,i+2])), (hcat(model_on_9x9[:,:,1,i+3], model_on_9x9[:,:,1,i+4], model_on_9x9[:,:,1,i+5])), (hcat(model_on_9x9[:,:,1,i+6], model_on_9x9[:,:,1,i+7], model_on_9x9[:,:,1,i+8])))
    push!(things, thing)
end
things
things[1]
things[2]







m1 = model_on_9x9[1][:,:,1,1]
m2 = model_on_9x9[1][:,:,1,2]
m3 = model_on_9x9[1][:,:,1,3]
row1 = hcat(m1,m2,m3)

m4 = model_on_9x9[1][:,:,1,4]
m5 = model_on_9x9[1][:,:,1,5]
m6 = model_on_9x9[1][:,:,1,6]
row2 = hcat(m4,m5,m6)

m7 = model_on_9x9[1][:,:,1,7]
m8 = model_on_9x9[1][:,:,1,8]
m9 = model_on_9x9[1][:,:,1,9]
row3 = hcat(m7,m8,m9)

map27x27 = vcat(row1,row2,row3)
testmap1 = heatmap(map27x27)

#connectivity map (true map -- not run with model)
truem1 = nine_nine[1][2][:,:,1,1]
truem2 = nine_nine[1][2][:,:,1,2]
truem3 = nine_nine[1][2][:,:,1,3]
truerow1 = hcat(truem1,truem2,truem3)

truem4 = nine_nine[1][2][:,:,1,7]
truem5 = nine_nine[1][2][:,:,1,8]
truem6 = nine_nine[1][2][:,:,1,9]
truerow2 = hcat(truem4,truem5,truem6)
heatmap(truerow2)

truem7 = nine_nine[3][2][:,:,1,22]
truem8 = nine_nine[3][2][:,:,1,23]
truem9 = nine_nine[3][2][:,:,1,24]
truerow3 = hcat(truem7,truem8,truem9)
heatmap(truerow3)

truemap27x27 = vcat(truerow1, truerow2, truerow3)
truemap1 = heatmap(truemap27x27)

# plot(truemap1, truemap1_2)

comparemap1 = heatmap(validate_connect27x27[1]) #want to compare to the true connectivity map at the same index points

refmaps = plot(truemap1, comparemap1) #bottom 3 match... the rest don't. WHY

scatterplotmaps = scatter(map27x27, truemap27x27, leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed", yaxis="predicted")
plot(testmap1, truemap1, comparemap1, scatterplotmaps)













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

maps9x9: vector of n elements of dims 9x9x2
connect9x9: vector of n elements of dims 9x9

check accuracy with:
validation_maps: vector of m elements of dims 27x27x2
validation_connect: vector of m elements of dims 27x27

=#

model_on_9x9_2 = model(nine_nine_2[1][1])


collect(Iterators.partition(model_on_9x9_2, 3))



m1_2 = model_on_9x9_2[:,:,1,1]
m2_2 = model_on_9x9_2[:,:,1,2]
m3_2 = model_on_9x9_2[:,:,1,3]
row1_2 = hcat(m1_2,m2_2,m3_2)

m4_2 = model_on_9x9_2[:,:,1,4]
m5_2 = model_on_9x9_2[:,:,1,5]
m6_2 = model_on_9x9_2[:,:,1,6]
row2_2 = hcat(m4_2,m5_2,m6_2)

m7_2 = model_on_9x9_2[:,:,1,7]
m8_2 = model_on_9x9_2[:,:,1,8]
m9_2 = model_on_9x9_2[:,:,1,9]
row3_2 = hcat(m7_2,m8_2,m9_2)

map27x27_2 = vcat(row1_2,row2_2,row3_2)
testmap1_2 = heatmap(map27x27_2)

#connectivity map (true map -- not run with model)
truem1 = nine_nine[1][2][:,:,1,1]
truem2 = nine_nine[1][2][:,:,1,2]
truem3 = nine_nine[1][2][:,:,1,3]
truerow1 = hcat(truem1,truem2,truem3)

truem4 = nine_nine[1][2][:,:,1,4]
truem5 = nine_nine[1][2][:,:,1,5]
truem6 = nine_nine[1][2][:,:,1,6]
truerow2 = hcat(truem4,truem5,truem6)

truem7 = nine_nine[1][2][:,:,1,7]
truem8 = nine_nine[1][2][:,:,1,8]
truem9 = nine_nine[1][2][:,:,1,9]
truerow3 = hcat(truem7,truem8,truem9)

truemap27x27 = vcat(truerow1, truerow2, truerow3)
truemap1 = heatmap(truemap27x27)

comparemap1 = heatmap(compare_with_27x27[1][2][:,:,1,1]) #want to compare to the true connectivity map at the same index points

refmaps = plot(truemap1, comparemap1)

scatterplotmaps = scatter(map27x27, truemap27x27, leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed", yaxis="predicted")
plot(testmap1, truemap1, comparemap1, scatterplotmaps)
