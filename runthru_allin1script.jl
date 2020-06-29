#=
This script loads the original CSV datafiles and returns m nxn random samples for training and testing sets

Input:
  resistance.csv
  origin.csv
  connectivity.csv

Output:
  maps: x_train
  connect: y_train
  test_maps: x_test
  test_connect: y_test
=#


include("libraries.jl")
include("functions.jl")
include("model.jl")
# cd(@__DIR__)

#Read in the CSV (comma separated values) file and convert them to arrays.
Resistance = readasc("data/maps_for_Kiri/Resistance_zone_beta_OursNoir.asc"; nd="NODATA")
Origin = readasc("data/input/origin.asc"; nd="NODATA")
Connectivity = readasc("data/maps_for_Kiri/Current_OursNoir.asc")

nan_to_0(Resistance)
nan_to_0(Origin)
nan_to_0(Connectivity)

#create Training dataset
# Extract 150 random 9x9 resistance, origin, and connectivity layers
Stride = 9
batch_size = 32
Random.seed!(1234)

maps, connect, test_maps, test_connect = make_datasets(Resistance, Origin, Connectivity)

train_maps, train_connect, valid_maps, valid_connect = partition_dataset(maps, connect)

train_set, validation_set = make_sets(train_maps, train_connect, valid_maps, valid_connect)

#TODO: run train_model.jl


# Plot
p1 = heatmap(validation_set[1][2][:,:,1,32], title="predicted") #connectivity map
p2 = heatmap(model(validation_set[1][1])[:,:,1,32], title="observed") #resistance and origin layer map
p3 = scatter(validation_set[1][2][:,:,1,32], model(validation_set[1][1])[:,:,1,32], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
plot(p1,p2,p3)
savefig("figures/fullblackbear_$(run)sec_$(best_acc*100)%.png")





#### For full map ####

Stride = 9
Desired_x = Int(size(Connectivity,2)/9)
rem_desired = rem(size(Connectivity, 1), Stride)
Desired_y = Int((size(Connectivity, 1)-rem_desired)/9)

c = Connectivity[1:end-rem_desired, :]
r = Resistance[1:end-rem_desired, :]
o = Origin[1:end-rem_desired, :]

#get coordinates for full connectivity map
all_coord = []
for i in CartesianIndices(c)
  coords = i
  push!(all_coord, coords)
end
all_coord = Tuple.(all_coord)


#create range around first coordinate
first_coor = first(all_coord)
tup1, tup2 = Tuple(first_coor)
range = [tup1:tup1+(size(c,2))-1, tup2:tup2+(size(c,1))-1]

#get every single index in samples
x_idx = collect(range[2])
y_idx = collect(range[1])

#get the first coordinate for each smaller (9x9) sample
x_idxes = x_idx[1:Stride:end]
y_idxes = y_idx[1:Stride:end]

#get the 9 starting coordinates
replicate_x = repeat(x_idxes, inner = length(y_idxes))
replicate_y = repeat(y_idxes, outer = length(x_idxes))

#zip coordinates together
zip_coor = Tuple.(zip(replicate_x, replicate_y))
last(zip_coor)

#create 9x9 samples
maps9x9 = []
connect9x9 = []
for (i,j) in zip_coor
  x_res = r[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3)
  y = c[i:(i+Stride-1),j:(j+Stride-1)]
  push!(maps9x9, x)
  push!(connect9x9, y)
end


batch_size=32
### minibatch ###
#subtract remainders to ensure all minibatches are the same length
droplast9x9 = rem(length(maps9x9), batch_size)
mb_idxs9x9 = Iterators.partition(1:length(maps9x9)-droplast9x9, batch_size)
#train set in the form of batches
nine_nine = [make_minibatch(maps9x9, connect9x9, i) for i in mb_idxs9x9]



### verify connectivity values are the same ###
truemap = [reduce(hcat, p) for p in Iterators.partition(connect9x9, Desired_x)]
truemap = [reduce(vcat, p) for p in Iterators.partition(truemap, Desired_y)]
# heatmap(truemap_fox[1])
all(isapprox.(c, truemap[1]))


# @time include("model.jl")
@time @load "BSON/full_blackbear_849sec_9890acc.bson" params #upload last saved model
Flux.loadparams!(model, params) #new model will now be identical to the one saved params for

##### Run model on data #####
#run trained model on new minibatched data (from )
model_on_9x9 = trained_model(nine_nine)

#if less than 0, = 0; if >1 = 1
model_on_9x9_zero = replace.(x -> x < 0 ? 0 : x, model_on_9x9)
model_9x9 = replace.(x -> x > 1 ? 1 : x, model_on_9x9_zero)


#reduce 4D to 2D
mod = []
for t in model_9x9
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

originalmap = heatmap(c)

fullmap = heatmap(stitchedmap[1])
# savefig("figures/bear_model.png")

scat = scatter(stitchedmap[1], c[1:end-9, :], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
# savefig("figures/scatter_bear_model.png")

difference = stitchedmap[1] - c[1:end-9, :] #overestimating = 1; underestimating = -1
dif = heatmap(difference)
# savefig("figures/difference_bear_model.png")

plot(originalmap, fullmap, scat, dif)
# savefig("figures/comparisons_bear.png")
