@time include("libraries.jl")
@time include("functions.jl") #desired object found in line 23 of preprocess_idx.jl script
@time include("preprocess.jl")
@time include("validation_dataset.jl")
@time include("minibatch.jl")
@time include("model.jl")
@time @load "BSON/renard.bson" params #upload last saved model
Flux.loadparams!(model, params) #new model will now be identical to the one saved params for
# @time include("test_original_mod_renardroux.jl")


connectivity_renard = readasc("data/maps_for_Kiri/RR_cum_currmap.asc")
resistance_renard = readasc("data/maps_for_Kiri/Resistance_zone_beta_RR.asc"; nd="NODATA")
Origin = readasc("data/input/origin.asc"; nd="NODATA")

begin
    nan_to_0(connectivity_renard)
    nan_to_0(resistance_renard)
    nan_to_0(Origin)
end

Stride = 9
Desired_x = Int(size(connectivity_renard,2)/9)
rem_desired = rem(size(connectivity_renard, 1), Stride)
Desired_y = Int((size(connectivity_renard, 1)-rem_desired)/9)

c_fox = connectivity_renard[1:end-rem_desired, :]
r_fox = resistance_renard[1:end-rem_desired, :]
o_fox = Origin[1:end-rem_desired, :]

#get coordinates for full connectivity map
all_coord = []
for i in CartesianIndices(c_fox)
  coords = i
  push!(all_coord, coords)
end
all_coord = Tuple.(all_coord)

# #get starting coordinates for each 9x9
# y_coor = unique(first.(all_coord))[1:Stride:end]
# x_coor = unique(last.(all_coord))[1:Stride:end]
# y_start_coords = repeat(y_coor, outer = Desired_x)
# x_start_coords = repeat(x_coor, inner = Desired_y)

# coordinates_fox = []
# for i in 1:length(y_start_coords)
#   zip_renard = Tuple.(zip(y_start_coords[i], x_start_coords[i]))
#   push!(coordinates_fox, zip_renard)
# end
# last(coordinates_fox)

#create range around first coordinate
first_coor = first(all_coord)
tup1, tup2 = Tuple(first_coor)
range_fox = [tup1:tup1+(size(c_fox,2))-1, tup2:tup2+(size(c_fox,1))-1]

#get every single index in samples
x_idx_fox = collect(range_fox[2])
y_idx_fox = collect(range_fox[1])

#get the first coordinate for each smaller (9x9) sample
x_idxes_fox = x_idx_fox[1:Stride:end]
y_idxes_fox = y_idx_fox[1:Stride:end]

#get the 9 starting coordinates
replicate_x_fox = repeat(x_idxes_fox, inner = 134)
replicate_y_fox = repeat(y_idxes_fox, outer = 139)

#zip coordinates together
zip_fox = Tuple.(zip(replicate_x_fox, replicate_y_fox))
last(zip_fox)

#create 9x9 samples
maps9x9_fox = []
connect9x9_fox = []
for (i,j) in zip_fox
  x_res_fox = r_fox[i:(i+Stride-1),j:(j+Stride-1)]
  x_or_fox = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x_fox = cat(x_res_fox, x_or_fox, dims=3)
  y_fox = c_fox[i:(i+Stride-1),j:(j+Stride-1)]
  push!(maps9x9_fox, x_fox)
  push!(connect9x9_fox, y_fox)
end



### minibatch ###
#subtract remainders to ensure all minibatches are the same length
droplast9x9_fox = rem(length(maps9x9_fox), batch_size)
mb_idxs9x9_fox = Iterators.partition(1:length(maps9x9_fox)-droplast9x9_fox, batch_size)
#train set in the form of batches
nine_nine_fox = [make_minibatch(maps9x9_fox, connect9x9_fox, i) for i in mb_idxs9x9_fox]



### verify connectivity values are the same ###
truemap = [reduce(hcat, p) for p in Iterators.partition(connect9x9_fox, Desired_x)]
truemap_fox = [reduce(vcat, p) for p in Iterators.partition(truemap, Desired_y)]
heatmap(truemap_fox[1])




##### Run model on data #####
#run trained model on new minibatched data (from )
model_on_9x9_fox = trained_model(nine_nine_fox)


#reduce 4D to 2D
mod = []
for t in model_on_9x9_fox
  tmp2 = [t[:,:,1,i] for i in 1:batch_size]
  push!(mod, tmp2)
end
#reduce to one vector of arrays
mod = reduce(vcat, mod)

# remove_last = rem(length(mod), 9)
#hcat groups of three
stitched = [reduce(hcat, p) for p in Iterators.partition(mod, Desired_x)]
#vcat the stitched hcats
stitchedmap = [reduce(vcat, p) for p in Iterators.partition(stitched[1:end-1], 139)]

heatmap(stitchedmap[1])

s1 = scatter(mod[12100], connect9x9_fox[12100], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
p1 = heatmap(mod[12100])
p2 = heatmap(connect9x9_fox[12100])
plot(p1,p2,s1)
