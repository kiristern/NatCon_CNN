
@time @load "BSON/FIRST_ORIGINAL.bson" params #upload last saved model
Flux.loadparams!(model, params) #new model will now be identical to the one saved params for

#select new coordinates for obtaining n stridexstride samples
samples = 50
desired = 5 #how many 9x9 you want
stride = Stride*desired

#sample between these values only (to avoid going out of bounds)
lx = length(connectivity_carcajou[:,1])-stride
ly = length(connectivity_carcajou[1,:])-stride

Random.seed!(1234)
#select coordinates where there is data
cart_idx = sample(findall(connectivity_carcajou[1:lx, 1:ly] .> 0), samples)
coordinates = Tuple.(cart_idx)

#create range around each sample index (length of stride)
range = []
for i in cart_idx
  a, b = Tuple(i)
  c = [a:a+stride-1,b:b+stride-1]
  push!(range, c)
end

#make 45x45 imgs from coordinates as reference to compare
valid_connect_map_carcajou = []
for i in coordinates
  y = connectivity_carcajou[first(i):first(i)+stride-1,last(i):last(i)+stride-1] #matrix we want to predict
    push!(valid_connect_map_carcajou, y)
end
valid_connect_map_carcajou

#get every single index in samples
x_indices = []
y_indices = []
for i in 1:length(range)
  x_idx = collect(range[i][1])
  y_idx = collect(range[i][2])
  push!(x_indices, [x_idx...])
  push!(y_indices, [y_idx...])
end
x_indices
y_indices

#get the first coordinate for each smaller (9x9) sample
x_idxes = [x[1:Stride:end] for x in x_indices]
y_idxes = [y[1:Stride:end] for y in y_indices]

#get the 9 starting coordinates
replicate_x = repeat.(x_idxes, inner = desired)
replicate_y = repeat.(y_idxes, outer = desired)

#zip coordinates together
dup_coor = []
for i in 1:length(replicate_x)
  zip_dup = Tuple.(zip(replicate_x[i], replicate_y[i])) #y_idxes from preprocess_idx.jl script
  push!(dup_coor, zip_dup)
end
dup_coor

#create 9x9 samples
maps9x9_carcajou = []
connect9x9_carcajou = []
for (i,j) in reduce(vcat, dup_coor)
  x_res2 = resistance_carcajou[i:(i+Stride-1),j:(j+Stride-1)]
  x_or2 = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x2 = cat(x_res2, x_or2, dims=3)
  y2 = connectivity_carcajou[i:(i+Stride-1),j:(j+Stride-1)]
  push!(maps9x9_carcajou, x2)
  push!(connect9x9_carcajou, y2)
end
maps9x9_carcajou
connect9x9_carcajou



### minibatch ###
#subtract remainders to ensure all minibatches are the same length
droplast9x9 = rem(length(maps9x9_carcajou), batch_size)
mb_idxs9x9 = Iterators.partition(1:length(maps9x9_carcajou)-droplast9x9, batch_size)
#train set in the form of batches
nine_nine_carcajou = [make_minibatch(maps9x9_carcajou, connect9x9_carcajou, i) for i in mb_idxs9x9]
nine_nine_carcajou



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
  tmp2 = [t[:,:,1,i] for i in 1:batch_size]
  push!(minib_carcajou, tmp2)
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
scatterplotmaps_carcajou = scatter(stitchedmap_carcajou[20], valid_connect_map_carcajou[20], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
plot(heatmap(stitchedmap_carcajou[20]), heatmap(valid_connect_map_carcajou[20]), scatterplotmaps_carcajou)
savefig("figures/original_trained_model_carcajou[20].png")
