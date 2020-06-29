
@time @load "BSON/FIRST_ORIGINAL.bson" params #upload last saved model
Flux.loadparams!(model, params) #new model will now be identical to the one saved params for

#select new coordinates for obtaining n stridexstride samples
samples = 100
desired = 5 #how many 9x9 you want
stride = Stride*desired

#sample between these values only (to avoid going out of bounds)
lx_cougar = length(connectivity_cougar[:,1])-stride
ly_cougar = length(connectivity_cougar[1,:])-stride

Random.seed!(1234)
#select coordinates where there is data
cart_idx_cougar = sample(findall(connectivity_cougar[1:lx_cougar, 1:ly_cougar] .> 0), samples)
coordinates_cougar = Tuple.(cart_idx_cougar)

#create range around each sample index (length of stride)
range_cougar = []
for i in cart_idx_cougar
  a_cougar, b_cougar = Tuple(i)
  c_cougar = [a_cougar:a_cougar+stride-1,b_cougar:b_cougar+stride-1]
  push!(range_cougar, c_cougar)
end

#make 45x45 imgs from coordinates as reference to compare
valid_connect_map_cougar = []
for i in coordinates_cougar
  y_cougar = connectivity_cougar[first(i):first(i)+stride-1,last(i):last(i)+stride-1] #matrix we want to predict
    push!(valid_connect_map_cougar, y_cougar)
end

#get every single index in samples
x_indices_cougar = []
y_indices_cougar = []
for i in 1:length(range_cougar)
  x_idx_cougar = collect(range_cougar[i][1])
  y_idx_cougar = collect(range_cougar[i][2])
  push!(x_indices_cougar, [x_idx_cougar...])
  push!(y_indices_cougar, [y_idx_cougar...])
end

#get the first coordinate for each smaller (9x9) sample
x_idxes_cougar = [x[1:Stride:end] for x in x_indices_cougar]
y_idxes_cougar = [y[1:Stride:end] for y in y_indices_cougar]

#get the 9 starting coordinates
replicate_x_cougar = repeat.(x_idxes_cougar, inner = desired)
replicate_y_cougar = repeat.(y_idxes_cougar, outer = desired)

#zip coordinates together
dup_coor_cougar = []
for i in 1:length(replicate_x_cougar)
  zip_dup_cougar = Tuple.(zip(replicate_x_cougar[i], replicate_y_cougar[i])) #y_idxes from preprocess_idx.jl script
  push!(dup_coor_cougar, zip_dup_cougar)
end

#create 9x9 samples
maps9x9_cougar = []
connect9x9_cougar = []
for (i,j) in reduce(vcat, dup_coor_cougar)
  x_res_cougar = resistance_cougar[i:(i+Stride-1),j:(j+Stride-1)]
  x_or_cougar = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x_cougar = cat(x_res_cougar, x_or_cougar, dims=3)
  y_cougar = connectivity_cougar[i:(i+Stride-1),j:(j+Stride-1)]
  push!(maps9x9_cougar, x_cougar)
  push!(connect9x9_cougar, y_cougar)
end




### minibatch ###
#subtract remainders to ensure all minibatches are the same length
droplast9x9_cougar = rem(length(maps9x9_cougar), batch_size)
mb_idxs9x9_cougar = Iterators.partition(1:length(maps9x9_cougar)-droplast9x9_cougar, batch_size)
#train set in the form of batches
nine_nine_cougar = [make_minibatch(maps9x9_cougar, connect9x9_cougar, i) for i in mb_idxs9x9_cougar]



### verify connectivity values are the same ###
#stitch together 3 (9x9) x 3 (9x9)
truemap_cougar = stitch2d(connect9x9_cougar)
plot(heatmap(truemap_cougar[1]), heatmap(valid_connect_map_cougar[1]))

#compare connectivity layers from minibatching
mini_truemap_cougar = stitch4d([nine_nine_cougar[i][2] for i in eachindex(nine_nine_cougar)])

#compare all connectivity layers
plot(heatmap(truemap_cougar[1]), heatmap(valid_connect_map_cougar[1]), heatmap(mini_truemap_cougar[1]))

### verify non-visually ###
#reduce 4D to 2D
minib_cougar = []
for t in [nine_nine_cougar[i][2] for i in eachindex(nine_nine_cougar)] #for t in each connectivity layer in nine_nine
  tmp_cougar = [t[:,:,1,i] for i in 1:batch_size]
  push!(minib_cougar, tmp_cougar)
end
#reduce to single vector
minib_cougar = reduce(vcat, minib_cougar)
#check if connectivity of minibatch values are the same as connect
all(isapprox.(minib_cougar, connect9x9_cougar[1:length(minib_cougar)]))

#run trained model on new minibatched data (from )
model_on_9x9_cougar = trained_model(nine_nine_cougar)

#stitch together 27x27 maps
stitchedmap_cougar = stitch4d(model_on_9x9_cougar)

#plot
scatterplotmaps_cougar = scatter(stitchedmap_cougar[1], valid_connect_map_cougar[1], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
plot(heatmap(stitchedmap_cougar[1]), heatmap(valid_connect_map_cougar[1]), scatterplotmaps_cougar)
savefig("figures/original_trained_model_cougar[1].png")
