#=

Get a stridexstride sample from original data maps and, using the trained (working) model, predict nine 9x9 images and "stitch" together in desiredxdesired batch to get a final stridexstride image.

n desiredxdesired images; aka, groups of "desired" (9x9) by "desired" (9x9) images
Output:
  maps9x9: vector of n elements of dims 9x9x2
  connect9x9: vector of n elements of dims 9x9
  validation_maps: vector of m elements of dims 27x27x2
  validation_connect: vector of m elements of dims 27x27

=#


# cd(@__DIR__)
#
# @time include("libraries.jl")
# @time include("functions.jl")
# @time include("preprocess.jl")

#select new coordinates for obtaining n stridexstride samples
samples = 100
desired = 5 #how many 9x9 you want
stride = Stride*desired

#sample between these values only (to avoid going out of bounds)
lx = length(Resistance[:,1])-stride
ly = length(Resistance[1,:])-stride

Random.seed!(1234)
#select coordinates where there is data
cart_idx = sample(findall(Connectivity[1:lx, 1:ly] .> 0), samples)
coordinates = Tuple.(cart_idx)

#create range around each sample index (length of stride)
range = []
for i in cart_idx
  a, b = Tuple(i)
  c = [a:a+stride-1,b:b+stride-1]
  push!(range, c)
end

#make 27x27 imgs from coordinates as reference to compare
validation_connectivity_map = []
for i in coordinates
  y = Connectivity[first(i):first(i)+stride-1,last(i):last(i)+stride-1] #matrix we want to predict
    push!(validation_connectivity_map, y)
end
validation_connectivity_map

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
maps9x9 = []
connect9x9 = []
for (i,j) in reduce(vcat, dup_coor)
  x_res2 = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  x_or2 = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x2 = cat(x_res2, x_or2, dims=3)
  y2 = Connectivity[i:(i+Stride-1),j:(j+Stride-1)]
  push!(maps9x9, x2)
  push!(connect9x9, y2)
end
maps9x9
connect9x9



### minibatch ###
#subtract remainders to ensure all minibatches are the same length
droplast9x9 = rem(length(maps9x9), batch_size)
mb_idxs9x9 = Iterators.partition(1:length(maps9x9)-droplast9x9, batch_size)
#train set in the form of batches
nine_nine = [make_minibatch(maps9x9, connect9x9, i) for i in mb_idxs9x9]
nine_nine



### verify connectivity values are the same ###
#stitch together 3 (9x9) x 3 (9x9)
truemap = stitch2d(connect9x9)
plot(heatmap(truemap[50]), heatmap(validation_connectivity_map[50]))

#compare connectivity layers from minibatching
mini_truemap = stitch4d([nine_nine[i][2] for i in eachindex(nine_nine)])

#compare all connectivity layers
plot(heatmap(truemap[56]), heatmap(validation_connectivity_map[56]), heatmap(mini_truemap[56]))

### verify non-visually ###
#reduce 4D to 2D
minib = []
for t in [nine_nine[i][2] for i in eachindex(nine_nine)] #for t in each connectivity layer in nine_nine
  tmp2 = [t[:,:,1,i] for i in 1:batch_size]
  push!(minib, tmp2)
end
#reduce to single vector
minib = reduce(vcat, minib)
#check if connectivity of minibatch values are the same as connect
all(isapprox.(minib, connect9x9[1:length(minib)]))
