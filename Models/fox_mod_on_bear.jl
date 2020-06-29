@time include("libraries.jl")
@time include("functions.jl") #desired object found in line 23 of preprocess_idx.jl script


connectivity_bear = readasc("data/maps_for_Kiri/Current_OursNoir.asc")
resistance_bear = readasc("data/maps_for_Kiri/Resistance_zone_beta_OursNoir.asc"; nd="NODATA")
Origin = readasc("data/input/origin.asc"; nd="NODATA")

begin
    nan_to_0(connectivity_bear)
    nan_to_0(resistance_bear)
    nan_to_0(Origin)
end

Stride = 9
Desired_x = Int(size(connectivity_bear,2)/9)
rem_desired = rem(size(connectivity_bear, 1), Stride)
Desired_y = Int((size(connectivity_bear, 1)-rem_desired)/9)

c_bear = connectivity_bear[1:end-rem_desired, :]
r_bear = resistance_bear[1:end-rem_desired, :]
o_bear = Origin[1:end-rem_desired, :]

#get coordinates for full connectivity map
all_coord = []
for i in CartesianIndices(c_bear)
  coords = i
  push!(all_coord, coords)
end
all_coord = Tuple.(all_coord)


#create range around first coordinate
first_coor = first(all_coord)
tup1, tup2 = Tuple(first_coor)
range_bear = [tup1:tup1+(size(c_bear,2))-1, tup2:tup2+(size(c_bear,1))-1]

#get every single index in samples
x_idx_bear = collect(range_bear[2])
y_idx_bear = collect(range_bear[1])

#get the first coordinate for each smaller (9x9) sample
x_idxes_bear = x_idx_bear[1:Stride:end]
y_idxes_bear = y_idx_bear[1:Stride:end]

#get the 9 starting coordinates
replicate_x_bear = repeat(x_idxes_bear, inner = 134)
replicate_y_bear = repeat(y_idxes_bear, outer = 139)

#zip coordinates together
zip_bear = Tuple.(zip(replicate_x_bear, replicate_y_bear))
last(zip_bear)

#create 9x9 samples
maps9x9_bear = []
connect9x9_bear = []
for (i,j) in zip_bear
  x_res_bear = r_bear[i:(i+Stride-1),j:(j+Stride-1)]
  x_or_bear = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x_bear = cat(x_res_bear, x_or_bear, dims=3)
  y_bear = c_bear[i:(i+Stride-1),j:(j+Stride-1)]
  push!(maps9x9_bear, x_bear)
  push!(connect9x9_bear, y_bear)
end


batch_size=32
### minibatch ###
#subtract remainders to ensure all minibatches are the same length
droplast9x9_bear = rem(length(maps9x9_bear), batch_size)
mb_idxs9x9_bear = Iterators.partition(1:length(maps9x9_bear)-droplast9x9_bear, batch_size)
#train set in the form of batches
nine_nine_bear = [make_minibatch(maps9x9_bear, connect9x9_bear, i) for i in mb_idxs9x9_bear]



### verify connectivity values are the same ###
truemap = [reduce(hcat, p) for p in Iterators.partition(connect9x9_bear, Desired_x)]
truemap_bear = [reduce(vcat, p) for p in Iterators.partition(truemap, Desired_y)]
heatmap_original = heatmap(truemap_bear[1])
all(isapprox.(c_bear, truemap_bear[1]))




@time include("model.jl")
@time @load "BSON/fox_sliding_window.bson" params #upload last saved model
Flux.loadparams!(model, params)

##### Run model on data #####
#run trained model on new minibatched data (from )
model_on_9x9_bear = trained_model(nine_nine_bear)

#if less than 0, = 0; if >1 = 1
model_on_9x9_zero = replace.(x -> x < 0 ? 0 : x, model_on_9x9_bear)
model_9x9_bear = replace.(x -> x > 1 ? 1 : x, model_on_9x9_zero)


#reduce 4D to 2D
mod = []
for t in model_9x9_bear
  tmp2 = [t[:,:,1,i] for i in 1:batch_size]
  push!(mod, tmp2)
end
#reduce to one vector of arrays
mod = reduce(vcat, mod)

# remove_last = rem(length(mod), 9)
#hcat groups of three
stitched_bear = [reduce(hcat, p) for p in Iterators.partition(mod, Desired_x)]
#vcat the stitched hcats
stitchedmap_bear = [reduce(vcat, p) for p in Iterators.partition(stitched_bear[1:end-1], 139)]

heatmap_model = heatmap(stitchedmap_bear[1])


scatterplot = scatter(stitchedmap_bear[1], c_bear[1:end-9, :], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")

difference = heatmap(stitchedmap_bear[1] - c_bear[1:end-9, :]) #overestimating = 1; underestimating = -1


plot(heatmap_original, heatmap_model, difference, scatterplot)
savefig("figures/fox_on_bear_plots.png")
