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


@time include("../libraries.jl")
@time include("../functions.jl")
include("species.jl")
# cd(@__DIR__)


#create Training dataset
# Extract 150 random 9x9 resistance, origin, and connectivity layers
Random.seed!(1234)
Stride = 9
batch_size=32

maps_multisp, connect_multisp, test_multisp, test_maps_connect_multisp = [], [], [], []

push!(maps_multisp, make_datasets(resistance_carcajou, Origin, connectivity_carcajou)[1])
push!(connect_multisp, make_datasets(resistance_carcajou, Origin, connectivity_carcajou)[2])
push!(test_multisp, make_datasets(resistance_carcajou, Origin, connectivity_carcajou)[3])
push!(test_maps_connect_multisp, make_datasets(resistance_carcajou, Origin, connectivity_carcajou)[4])

push!(maps_multisp, make_datasets(resistance_ours, Origin, connectivity_ours)[1])
push!(connect_multisp, make_datasets(resistance_ours, Origin, connectivity_ours)[2])
push!(test_multisp, make_datasets(resistance_ours, Origin, connectivity_ours)[3])
push!(test_maps_connect_multisp, make_datasets(resistance_ours, Origin, connectivity_ours)[4])

push!(maps_multisp, make_datasets(resistance_cougar, Origin, connectivity_cougar)[1])
push!(connect_multisp, make_datasets(resistance_cougar, Origin, connectivity_cougar)[2])
push!(test_multisp, make_datasets(resistance_cougar, Origin, connectivity_cougar)[3])
push!(test_maps_connect_multisp, make_datasets(resistance_cougar, Origin, connectivity_cougar)[4])

push!(maps_multisp, make_datasets(resistance_renard, Origin, connectivity_renard)[1])
push!(connect_multisp, make_datasets(resistance_renard, Origin, connectivity_renard)[2])
push!(test_multisp, make_datasets(resistance_renard, Origin, connectivity_renard)[3])
push!(test_maps_connect_multisp, make_datasets(resistance_renard, Origin, connectivity_renard)[4])

push!(maps_multisp, make_datasets(resistance_ratonlaveur, Origin, connectivity_ratonlaveur)[1])
push!(connect_multisp, make_datasets(resistance_ratonlaveur, Origin, connectivity_ratonlaveur)[2])
push!(test_multisp, make_datasets(resistance_ratonlaveur, Origin, connectivity_ratonlaveur)[3])
push!(test_maps_connect_multisp, make_datasets(resistance_ratonlaveur, Origin, connectivity_ratonlaveur)[4])

maps_multisp
connect_multisp
test_multisp
test_maps_connect_multisp

maps_multisp = vcat(maps_multisp...)
connect_multisp = vcat(connect_multisp...)


#script returns:
# maps_multisp
# connect_multisp
# test_maps_multisp
# test_connect_multisp


train_maps_multisp, train_connect_multisp, valid_maps_multisp, valid_connect_multisp = partition_dataset(maps_multisp, connect_multisp)

train_set_multisp, validation_set_multisp = make_sets(train_maps_multisp, train_connect_multisp, valid_maps_multisp, valid_connect_multisp)

#TODO: run on train_model.jl script! :D

# Plot
p1 = heatmap(validation_set_multisp[1][2][:,:,1,32], title="predicted") #connectivity map
p2 = heatmap(model(validation_set_multisp[1][1])[:,:,1,32], title="observed") #resistance and origin layer map
p3 = scatter(validation_set_multisp[1][2][:,:,1,32], model(validation_set_multisp[1][1])[:,:,1,32], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
plot(p1,p2,p3)
savefig("figures/_allsp_$(run)sec_$(best_acc*100)%_$(last_improvement)epoch.png")






#Read in the CSV (comma separated values) file and convert them to arrays.
Resistance = readasc("data/maps_for_Kiri/Resistance_zone_beta_Carcajou.asc"; nd="NODATA")
Origin = readasc("data/input/origin.asc"; nd="NODATA")
Connectivity = readasc("data/maps_for_Kiri/Current_Carcajou.asc")

begin
  nan_to_0(Resistance)
  nan_to_0(Origin)
  nan_to_0(Connectivity)
end



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
Range = [tup1:tup1+(size(c,2))-1, tup2:tup2+(size(c,1))-1]

#get every single index in samples
x_idx = collect(Range[2])
y_idx = collect(Range[1])

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


@time include("../model.jl")
@time @load "BSON/allspecies_3033sec_9842acc_atom.bson" params #upload last saved model
Flux.loadparams!(model, params) #new model will now be identical to the one saved params for

##### Run model on data #####
#run trained model on new minibatched data (from )
@time model_on_9x9 = trained_model(nine_nine)

#if less than 0, = 0
model_9x9 = replace.(x -> x < 0 ? 0 : x, model_on_9x9)

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
stitchedmap = [reduce(vcat, p) for p in Iterators.partition(stitched[1:end-1], length(stitched))]

minimum(stitchedmap[1])
maximum(stitchedmap[1])

#normalize data between 0-1 while keeping weights
map_scale = (stitchedmap[1] .- minimum(stitchedmap[1])) ./ (maximum(map_scale) .- minimum(map_scaled))

minimum(map_scale)
maximum(map_scale)

#MWE
# X = [0.0 -0.5 0.5; 0.0 1.0 2.0]
# standardize(UnitRangeTransform, X, dims=2) #doesn't work because it's across rows and not entire array
# Xstd = X .- minimum(X)
# Xstd = Xstd ./ (maximum(Xstd) .- minimum(Xstd))


originalmap = heatmap(c)
savefig("figures/original_carcajou.pdf")

fullmap = heatmap(map_scale)
savefig("figures/dense9layer_allsp_on_carcajou.png")

scat = scatter(stitchedmap[1], c[1:end-9, :], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
savefig("figures/scatter_dense9layer_allsp_on_carcajou.png")

difference = stitchedmap[1] - c[1:end-9, :] #overestimating = 1; underestimating = -1
dif = heatmap(difference)
savefig("figures/difference_dense9layer_allsp_on_carcajou.png")
