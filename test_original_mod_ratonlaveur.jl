
@time @load "BSON/FIRST_ORIGINAL.bson" params #upload last saved model
Flux.loadparams!(model, params) #new model will now be identical to the one saved params for

#select new coordinates for obtaining n stridexstride samples
samples = 100
desired = 5 #how many 9x9 you want
stride = Stride*desired

#sample between these values only (to avoid going out of bounds)
lx_ratonlaveur = length(connectivity_ratonlaveur[:,1])-stride
ly_ratonlaveur = length(connectivity_ratonlaveur[1,:])-stride

Random.seed!(1234)
#select coordinates where there is data
cart_idx_ratonlaveur = sample(findall(connectivity_ratonlaveur[1:lx_ratonlaveur, 1:ly_ratonlaveur] .> 0), samples)
coordinates_ratonlaveur = Tuple.(cart_idx_ratonlaveur)

#create range around each sample index (length of stride)
range_ratonlaveur = []
for i in cart_idx_ratonlaveur
  a_ratonlaveur, b_ratonlaveur = Tuple(i)
  c_ratonlaveur = [a_ratonlaveur:a_ratonlaveur+stride-1,b_ratonlaveur:b_ratonlaveur+stride-1]
  push!(range_ratonlaveur, c_ratonlaveur)
end

#make 45x45 imgs from coordinates as reference to compare
valid_connect_map_ratonlaveur = []
for i in coordinates_ratonlaveur
  y_ratonlaveur = connectivity_ratonlaveur[first(i):first(i)+stride-1,last(i):last(i)+stride-1] #matrix we want to predict
    push!(valid_connect_map_ratonlaveur, y_ratonlaveur)
end

#get every single index in samples
x_indices_ratonlaveur = []
y_indices_ratonlaveur = []
for i in 1:length(range_ratonlaveur)
  x_idx_ratonlaveur = collect(range_ratonlaveur[i][1])
  y_idx_ratonlaveur = collect(range_ratonlaveur[i][2])
  push!(x_indices_ratonlaveur, [x_idx_ratonlaveur...])
  push!(y_indices_ratonlaveur, [y_idx_ratonlaveur...])
end

#get the first coordinate for each smaller (9x9) sample
x_idxes_ratonlaveur = [x[1:Stride:end] for x in x_indices_ratonlaveur]
y_idxes_ratonlaveur = [y[1:Stride:end] for y in y_indices_ratonlaveur]

#get the 9 starting coordinates
replicate_x_ratonlaveur = repeat.(x_idxes_ratonlaveur, inner = desired)
replicate_y_ratonlaveur = repeat.(y_idxes_ratonlaveur, outer = desired)

#zip coordinates together
dup_coor_ratonlaveur = []
for i in 1:length(replicate_x_ratonlaveur)
  zip_dup_ratonlaveur = Tuple.(zip(replicate_x_ratonlaveur[i], replicate_y_ratonlaveur[i])) #y_idxes from preprocess_idx.jl script
  push!(dup_coor_ratonlaveur, zip_dup_ratonlaveur)
end

#create 9x9 samples
maps9x9_ratonlaveur = []
connect9x9_ratonlaveur = []
for (i,j) in reduce(vcat, dup_coor_ratonlaveur)
  x_res_ratonlaveur = resistance_ratonlaveur[i:(i+Stride-1),j:(j+Stride-1)]
  x_or_ratonlaveur = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x_ratonlaveur = cat(x_res_ratonlaveur, x_or_ratonlaveur, dims=3)
  y_ratonlaveur = connectivity_ratonlaveur[i:(i+Stride-1),j:(j+Stride-1)]
  push!(maps9x9_ratonlaveur, x_ratonlaveur)
  push!(connect9x9_ratonlaveur, y_ratonlaveur)
end




### minibatch ###
#subtract remainders to ensure all minibatches are the same length
droplast9x9_ratonlaveur = rem(length(maps9x9_ratonlaveur), batch_size)
mb_idxs9x9_ratonlaveur = Iterators.partition(1:length(maps9x9_ratonlaveur)-droplast9x9_ratonlaveur, batch_size)
#train set in the form of batches
nine_nine_ratonlaveur = [make_minibatch(maps9x9_ratonlaveur, connect9x9_ratonlaveur, i) for i in mb_idxs9x9_ratonlaveur]



### verify connectivity values are the same ###
#stitch together 3 (9x9) x 3 (9x9)
truemap_ratonlaveur = stitch2d(connect9x9_ratonlaveur)
plot(heatmap(truemap_ratonlaveur[1]), heatmap(valid_connect_map_ratonlaveur[1]))

#compare connectivity layers from minibatching
mini_truemap_ratonlaveur = stitch4d([nine_nine_ratonlaveur[i][2] for i in eachindex(nine_nine_ratonlaveur)])

#compare all connectivity layers
plot(heatmap(truemap_ratonlaveur[1]), heatmap(valid_connect_map_ratonlaveur[1]), heatmap(mini_truemap_ratonlaveur[1]))

### verify non-visually ###
#reduce 4D to 2D
minib_ratonlaveur = []
for t in [nine_nine_ratonlaveur[i][2] for i in eachindex(nine_nine_ratonlaveur)] #for t in each connectivity layer in nine_nine
  tmp_ratonlaveur = [t[:,:,1,i] for i in 1:batch_size]
  push!(minib_ratonlaveur, tmp_ratonlaveur)
end
#reduce to single vector
minib_ratonlaveur = reduce(vcat, minib_ratonlaveur)
#check if connectivity of minibatch values are the same as connect
all(isapprox.(minib_ratonlaveur, connect9x9_ratonlaveur[1:length(minib_ratonlaveur)]))

#run trained model on new minibatched data (from )
model_on_9x9_ratonlaveur = trained_model(nine_nine_ratonlaveur)

#stitch together 27x27 maps
stitchedmap_ratonlaveur = stitch4d(model_on_9x9_ratonlaveur)

#plot
scatterplotmaps_ratonlaveur = scatter(stitchedmap_ratonlaveur[70], valid_connect_map_ratonlaveur[70], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
plot(heatmap(stitchedmap_ratonlaveur[70]), heatmap(valid_connect_map_ratonlaveur[70]), scatterplotmaps_ratonlaveur)
savefig("figures/original_trained_model_ratonlaveur[70].png")
