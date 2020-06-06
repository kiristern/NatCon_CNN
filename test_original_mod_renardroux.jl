
@time @load "BSON/FIRST_ORIGINAL.bson" params #upload last saved model
Flux.loadparams!(model, params) #new model will now be identical to the one saved params for

#select new coordinates for obtaining n stridexstride samples
samples = 100
desired = 5 #how many 9x9 you want
stride = Stride*desired

#sample between these values only (to avoid going out of bounds)
lx_renard = length(connectivity_renard[:,1])-stride
ly_renard = length(connectivity_renard[1,:])-stride

Random.seed!(1234)
#select coordinates where there is data
cart_idx_renard = sample(findall(connectivity_renard[1:lx_renard, 1:ly_renard] .> 0), samples)
coordinates_renard = Tuple.(cart_idx_renard)

#create range around each sample index (length of stride)
range_renard = []
for i in cart_idx_renard
  a_renard, b_renard = Tuple(i)
  c_renard = [a_renard:a_renard+stride-1,b_renard:b_renard+stride-1]
  push!(range_renard, c_renard)
end

#make 45x45 imgs from coordinates as reference to compare
valid_connect_map_renard = []
for i in coordinates_renard
  y_renard = connectivity_renard[first(i):first(i)+stride-1,last(i):last(i)+stride-1] #matrix we want to predict
    push!(valid_connect_map_renard, y_renard)
end

#get every single index in samples
x_indices_renard = []
y_indices_renard = []
for i in 1:length(range_renard)
  x_idx_renard = collect(range_renard[i][1])
  y_idx_renard = collect(range_renard[i][2])
  push!(x_indices_renard, [x_idx_renard...])
  push!(y_indices_renard, [y_idx_renard...])
end

#get the first coordinate for each smaller (9x9) sample
x_idxes_renard = [x[1:Stride:end] for x in x_indices_renard]
y_idxes_renard = [y[1:Stride:end] for y in y_indices_renard]

#get the 9 starting coordinates
replicate_x_renard = repeat.(x_idxes_renard, inner = desired)
replicate_y_renard = repeat.(y_idxes_renard, outer = desired)

#zip coordinates together
dup_coor_renard = []
for i in 1:length(replicate_x_renard)
  zip_dup_renard = Tuple.(zip(replicate_x_renard[i], replicate_y_renard[i])) #y_idxes from preprocess_idx.jl script
  push!(dup_coor_renard, zip_dup_renard)
end

#create 9x9 samples
maps9x9_renard = []
connect9x9_renard = []
for (i,j) in reduce(vcat, dup_coor_renard)
  x_res_renard = resistance_renard[i:(i+Stride-1),j:(j+Stride-1)]
  x_or_renard = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x_renard = cat(x_res_renard, x_or_renard, dims=3)
  y_renard = connectivity_renard[i:(i+Stride-1),j:(j+Stride-1)]
  push!(maps9x9_renard, x_renard)
  push!(connect9x9_renard, y_renard)
end




### minibatch ###
#subtract remainders to ensure all minibatches are the same length
droplast9x9_renard = rem(length(maps9x9_renard), batch_size)
mb_idxs9x9_renard = Iterators.partition(1:length(maps9x9_renard)-droplast9x9_renard, batch_size)
#train set in the form of batches
nine_nine_renard = [make_minibatch(maps9x9_renard, connect9x9_renard, i) for i in mb_idxs9x9_renard]



### verify connectivity values are the same ###
#stitch together 3 (9x9) x 3 (9x9)
truemap_renard = stitch2d(connect9x9_renard)
plot(heatmap(truemap_renard[1]), heatmap(valid_connect_map_renard[1]))

#compare connectivity layers from minibatching
mini_truemap_renard = stitch4d([nine_nine_renard[i][2] for i in eachindex(nine_nine_renard)])

#compare all connectivity layers
plot(heatmap(truemap_renard[1]), heatmap(valid_connect_map_renard[1]), heatmap(mini_truemap_renard[1]))

### verify non-visually ###
#reduce 4D to 2D
minib_renard = []
for t in [nine_nine_renard[i][2] for i in eachindex(nine_nine_renard)] #for t in each connectivity layer in nine_nine
  tmp_renard = [t[:,:,1,i] for i in 1:batch_size]
  push!(minib_renard, tmp_renard)
end
#reduce to single vector
minib_renard = reduce(vcat, minib_renard)
#check if connectivity of minibatch values are the same as connect
all(isapprox.(minib_renard, connect9x9_renard[1:length(minib_renard)]))

#run trained model on new minibatched data (from )
model_on_9x9_renard = trained_model(nine_nine_renard)

#stitch together 27x27 maps
stitchedmap_renard = stitch4d(model_on_9x9_renard)

#plot
scatterplotmaps_renard = scatter(stitchedmap_renard[70], valid_connect_map_renard[70], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
plot(heatmap(stitchedmap_renard[70]), heatmap(valid_connect_map_renard[70]), scatterplotmaps_renard)
savefig("figures/original_trained_model_renard[70].png")
