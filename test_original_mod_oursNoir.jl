
@time @load "BSON/FIRST_ORIGINAL.bson" params #upload last saved model
Flux.loadparams!(model, params) #new model will now be identical to the one saved params for

#select new coordinates for obtaining n stridexstride samples
samples = 100
desired = 5 #how many 9x9 you want
stride = Stride*desired

#sample between these values only (to avoid going out of bounds)
lx_ours = length(connectivity_ours[:,1])-stride
ly_ours = length(connectivity_ours[1,:])-stride

Random.seed!(1234)
#select coordinates where there is data
cart_idx_ours = sample(findall(connectivity_ours[1:lx_ours, 1:ly_ours] .> 0), samples)
coordinates_ours = Tuple.(cart_idx_ours)

#create range around each sample index (length of stride)
range_ours = []
for i in cart_idx_ours
  a_ours, b_ours = Tuple(i)
  c_ours = [a_ours:a_ours+stride-1,b_ours:b_ours+stride-1]
  push!(range_ours, c_ours)
end

#make 45x45 imgs from coordinates as reference to compare
valid_connect_map_ours = []
for i in coordinates_ours
  y_ours = connectivity_ours[first(i):first(i)+stride-1,last(i):last(i)+stride-1] #matrix we want to predict
    push!(valid_connect_map_ours, y_ours)
end

#get every single index in samples
x_indices_ours = []
y_indices_ours = []
for i in 1:length(range_ours)
  x_idx_ours = collect(range_ours[i][1])
  y_idx_ours = collect(range_ours[i][2])
  push!(x_indices_ours, [x_idx_ours...])
  push!(y_indices_ours, [y_idx_ours...])
end

#get the first coordinate for each smaller (9x9) sample
x_idxes_ours = [x[1:Stride:end] for x in x_indices_ours]
y_idxes_ours = [y[1:Stride:end] for y in y_indices_ours]

#get the 9 starting coordinates
replicate_x_ours = repeat.(x_idxes_ours, inner = desired)
replicate_y_ours = repeat.(y_idxes_ours, outer = desired)

#zip coordinates together
dup_coor_ours = []
for i in 1:length(replicate_x_ours)
  zip_dup_ours = Tuple.(zip(replicate_x_ours[i], replicate_y_ours[i])) #y_idxes from preprocess_idx.jl script
  push!(dup_coor_ours, zip_dup_ours)
end

#create 9x9 samples
maps9x9_ours = []
connect9x9_ours = []
for (i,j) in reduce(vcat, dup_coor_ours)
  x_res_ours = resistance_ours[i:(i+Stride-1),j:(j+Stride-1)]
  x_or_ours = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x_ours = cat(x_res_ours, x_or_ours, dims=3)
  y_ours = connectivity_ours[i:(i+Stride-1),j:(j+Stride-1)]
  push!(maps9x9_ours, x_ours)
  push!(connect9x9_ours, y_ours)
end




### minibatch ###
#subtract remainders to ensure all minibatches are the same length
droplast9x9_ours = rem(length(maps9x9_ours), batch_size)
mb_idxs9x9_ours = Iterators.partition(1:length(maps9x9_ours)-droplast9x9_ours, batch_size)
#train set in the form of batches
nine_nine_ours = [make_minibatch(maps9x9_ours, connect9x9_ours, i) for i in mb_idxs9x9_ours]



### verify connectivity values are the same ###
#stitch together 3 (9x9) x 3 (9x9)
truemap_ours = stitch2d(connect9x9_ours)
plot(heatmap(truemap_ours[1]), heatmap(valid_connect_map_ours[1]))

#compare connectivity layers from minibatching
mini_truemap_ours = stitch4d([nine_nine_ours[i][2] for i in eachindex(nine_nine_ours)])

#compare all connectivity layers
plot(heatmap(truemap_ours[1]), heatmap(valid_connect_map_ours[1]), heatmap(mini_truemap_ours[1]))

### verify non-visually ###
#reduce 4D to 2D
minib_ours = []
for t in [nine_nine_ours[i][2] for i in eachindex(nine_nine_ours)] #for t in each connectivity layer in nine_nine
  tmp_ours = [t[:,:,1,i] for i in 1:batch_size]
  push!(minib_ours, tmp_ours)
end
#reduce to single vector
minib_ours = reduce(vcat, minib_ours)
#check if connectivity of minibatch values are the same as connect
all(isapprox.(minib_ours, connect9x9_ours[1:length(minib_ours)]))

#run trained model on new minibatched data (from )
model_on_9x9_ours = trained_model(nine_nine_ours)

#stitch together 27x27 maps
stitchedmap_ours = stitch4d(model_on_9x9_ours)

#plot
scatterplotmaps_ours = scatter(stitchedmap_ours[10], valid_connect_map_ours[10], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
plot(heatmap(stitchedmap_ours[10]), heatmap(valid_connect_map_ours[10]), scatterplotmaps_ours)
savefig("figures/original_trained_model_ours[10].png")
