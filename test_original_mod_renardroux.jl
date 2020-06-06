
@time @load "BSON/FIRST_ORIGINAL.bson" params #upload last saved model
Flux.loadparams!(model, params) #new model will now be identical to the one saved params for

#select new coordinates for obtaining n stridexstride samples
samples = 100
desired = 5 #how many 9x9 you want
stride = Stride*desired

#sample between these values only (to avoid going out of bounds)
lx_carcajou = length(connectivity_carcajou[:,1])-stride
ly_carcajou = length(connectivity_carcajou[1,:])-stride

Random.seed!(1234)
#select coordinates where there is data
cart_idx_carcajou = sample(findall(connectivity_carcajou[1:lx_carcajou, 1:ly_carcajou] .> 0), samples)
coordinates_carcajou = Tuple.(cart_idx_carcajou)

#create range around each sample index (length of stride)
range_carcajou = []
for i in cart_idx_carcajou
  a_carcajou, b_carcajou = Tuple(i)
  c_carcajou = [a_carcajou:a_carcajou+stride-1,b_carcajou:b_carcajou+stride-1]
  push!(range_carcajou, c_carcajou)
end

#make 45x45 imgs from coordinates as reference to compare
valid_connect_map_carcajou = []
for i in coordinates_carcajou
  y_carcajou = connectivity_carcajou[first(i):first(i)+stride-1,last(i):last(i)+stride-1] #matrix we want to predict
    push!(valid_connect_map_carcajou, y_carcajou)
end

#get every single index in samples
x_indices_carcajou = []
y_indices_carcajou = []
for i in 1:length(range_carcajou)
  x_idx_carcajou = collect(range_carcajou[i][1])
  y_idx_carcajou = collect(range_carcajou[i][2])
  push!(x_indices_carcajou, [x_idx_carcajou...])
  push!(y_indices_carcajou, [y_idx_carcajou...])
end

#get the first coordinate for each smaller (9x9) sample
x_idxes_carcajou = [x[1:Stride:end] for x in x_indices_carcajou]
y_idxes_carcajou = [y[1:Stride:end] for y in y_indices_carcajou]

#get the 9 starting coordinates
replicate_x_carcajou = repeat.(x_idxes_carcajou, inner = desired)
replicate_y_carcajou = repeat.(y_idxes_carcajou, outer = desired)

#zip coordinates together
dup_coor_carcajou = []
for i in 1:length(replicate_x_carcajou)
  zip_dup_carcajou = Tuple.(zip(replicate_x_carcajou[i], replicate_y_carcajou[i])) #y_idxes from preprocess_idx.jl script
  push!(dup_coor_carcajou, zip_dup_carcajou)
end

#create 9x9 samples
maps9x9_carcajou = []
connect9x9_carcajou = []
for (i,j) in reduce(vcat, dup_coor_carcajou)
  x_res_carcajou = resistance_carcajou[i:(i+Stride-1),j:(j+Stride-1)]
  x_or_carcajou = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x_carcajou = cat(x_res_carcajou, x_or_carcajou, dims=3)
  y_carcajou = connectivity_carcajou[i:(i+Stride-1),j:(j+Stride-1)]
  push!(maps9x9_carcajou, x_carcajou)
  push!(connect9x9_carcajou, y_carcajou)
end




### minibatch ###
#subtract remainders to ensure all minibatches are the same length
droplast9x9_carcajou = rem(length(maps9x9_carcajou), batch_size)
mb_idxs9x9_carcajou = Iterators.partition(1:length(maps9x9_carcajou)-droplast9x9_carcajou, batch_size)
#train set in the form of batches
nine_nine_carcajou = [make_minibatch(maps9x9_carcajou, connect9x9_carcajou, i) for i in mb_idxs9x9_carcajou]



### verify connectivity values are the same ###
#stitch together 3 (9x9) x 3 (9x9)
truemap_carcajou = stitch2d(connect9x9_carcajou)
plot(heatmap(truemap_carcajou[1]), heatmap(valid_connect_map_carcajou[1]))

#compare connectivity layers from minibatching
mini_truemap_carcajou = stitch4d([nine_nine_carcajou[i][2] for i in eachindex(nine_nine_carcajou)])

#compare all connectivity layers
plot(heatmap(truemap_carcajou[1]), heatmap(valid_connect_map_carcajou[1]), heatmap(mini_truemap_carcajou[1]))

### verify non-visually ###
#reduce 4D to 2D
minib_carcajou = []
for t in [nine_nine_carcajou[i][2] for i in eachindex(nine_nine_carcajou)] #for t in each connectivity layer in nine_nine
  tmp_carcajou = [t[:,:,1,i] for i in 1:batch_size]
  push!(minib_carcajou, tmp_carcajou)
end
#reduce to single vector
minib_carcajou = reduce(vcat, minib_carcajou)
#check if connectivity of minibatch values are the same as connect
all(isapprox.(minib_carcajou, connect9x9_carcajou[1:length(minib_carcajou)]))

#run trained model on new minibatched data (from )
model_on_9x9_carcajou = trained_model(nine_nine_carcajou)

#stitch together 27x27 maps
stitchedmap_carcajou = stitch4d(model_on_9x9_carcajou)

#plot
scatterplotmaps_carcajou = scatter(stitchedmap_carcajou[70], valid_connect_map_carcajou[70], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
plot(heatmap(stitchedmap_carcajou[70]), heatmap(valid_connect_map_carcajou[70]), scatterplotmaps_carcajou)
savefig("figures/original_trained_model_carcajou[70].png")
