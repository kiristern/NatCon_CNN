#=

Get a 27x27 sample from original data maps and, using the trained (working) model, predict nine 9x9 images and "stitch" together in 3x3 batch to get a final 27x27 image.

n 27x27 images; aka, groups of 3 (9x9) by 3 (9x9) images
Output:
  maps9x9: vector of n elements of dims 9x9x2
  connect9x9: vector of n elements of dims 9x9
  validation_maps: vector of m elements of dims 27x27x2
  validation_connect: vector of m elements of dims 27x27

=#


# cd(@__DIR__)
#
@time include("libraries.jl")
@time include("functions.jl")
@time include("preprocess.jl")

#select new coordinates for obtaining n 27x27 samples
stride = 27
samples = 50

#TODO try to sample between these values (to avoid going out of bounds)
lx= length(Resistance[:,1])-stride
ly = length(Resistance[1,:])-stride

Random.seed!(1234)
#select coordinates where there is data
cart_idx = sample(findall(Connectivity .> 0), samples)
coordinates = Tuple.(cart_idx)

#create range around each sample index (length of stride)
range = []
for i in cart_idx
  a, b = Tuple(i)
  c = [a:a+stride-1,b:b+stride-1]
  push!(range, c)
end

#make 27x27 imgs from coordinates as reference to compare
validate_connect27x27 = []
for i in coordinates
  y = Connectivity[first(i):first(i)+stride-1,last(i):last(i)+stride-1] #matrix we want to predict
    push!(validate_connect27x27, y)
end
validate_connect27x27

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
#Tuple.(zip(x_idxes, y_idxes))

#get the 9 starting coordinates
replicate_x = repeat.(x_idxes, inner = 3)
replicate_y = repeat.(y_idxes, outer = 3)

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
for (i, j) in reduce(vcat, dup_coor)
  x_res2 = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  x_or2 = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x2 = cat(x_res2, x_or2, dims=3) #concatenate resistance and origin layers
  y2 = Connectivity[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  #if minimum(y2) > 0 #predict only when there is connectivity
  push!(maps9x9, x2)
  push!(connect9x9, y2)
end
maps9x9
connect9x9

validate_connect27x27


### minibatch ###
#subtract remainders to ensure all minibatches are the same length
droplast9x9 = rem(length(maps9x9), batch_size)
mb_idxs9x9 = Iterators.partition(1:length(maps9x9)-droplast9x9, batch_size)
#train set in the form of batches
nine_nine = [make_minibatch(maps9x9, connect9x9, i) for i in mb_idxs9x9]
nine_nine



#verify connectivity values are the same
truemap = [reduce(hcat, p) for p in Iterators.partition(connect9x9, 3)]
truemap = [reduce(vcat, p) for p in Iterators.partition(truemap, 3)]
plot(heatmap(truemap[1]), heatmap(validate_connect27x27[1]))

#compare minibatching: reduce from 4d to 2d
tmp = [nine_nine[i][2] for i in eachindex(nine_nine)]
m = []
for t in tmp
  tmp2 = [t[:,:,1,i] for i in 1:batch_size]
  push!(m, tmp2)
end
m
m = reduce(vcat, m)
mini_truemap = [reduce(hcat, p) for p in Iterators.partition(m, 3)]
mini_truemap = [reduce(vcat, p) for p in Iterators.partition(mini_truemap[1:149], 3)] #[1:149] because last element 9x9 not 9x27

#check if values are the same
all(isapprox.(m, connect9x9[1:448]))

plot(heatmap(truemap[1]), heatmap(validate_connect27x27[1]), heatmap(mini_truemap[1]))
