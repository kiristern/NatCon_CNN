#=

Get a 27x27 sample from original data maps and, using the trained (working) model, predict nine 9x9 images and "stitch" together in 3x3 batch to get a final 27x27 image.

Output:
    one 27x27 image; aka, 3 (9x9) by 3 (9x9) images

=#
cd(@__DIR__)

@time include("preprocess.jl")
@time include("validation_dataset.jl")
@time include("minibatch.jl")
@time include("train_model.jl")

using StatsBase

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

#create range around each sample point
range = []
for i in cart_idx
  a, b = Tuple(i)
  c = [a:a+stride-1,b:b+stride-1]
  push!(range, c)
end


#make 27x27 imgs from coordinates
validation_maps = []
validation_connect = []
for i in coordinates
  x_resistance = Resistance[first(i):first(i)+stride-1,last(i):last(i)+stride-1]
  x_origin = Origin[first(i):first(i)+stride-1,last(i):last(i)+stride-1]
  x = cat(x_resistance, x_origin, dims=3) #concatenate resistance and origin layers
  y = Connectivity[first(i):first(i)+stride-1,last(i):last(i)+stride-1] #matrix we want to predict
  if minimum(y) > 0 #predict only when there is connectivity
    push!(validation_maps, x)
    push!(validation_connect, y)
  end
end

[first(range[1][1]):Stride:last(range[1][1])+1]

#get every single index in samples
x_indices = []
y_indices = []
for i in 1:length(range)
  x_idx = collect(range[i][1])
  y_idx = collect(range[i][2])
  push!(x_indices, x_idx...)
  push!(y_indices, y_idx...)
end

#get the first coordinate for each smaller sample
x_idxes = x_indices[1:Stride:end]
y_idxes = y_indices[1:Stride:end]










#make a function get indices of each sample point
indices = []
CartesianIndices(res[])
findall(x->x==2, res[1])

for i in res, j in res
  idx = CartesianIndices(res[i][j])
  push!(indices, idx)
end

#from 27x27 samples, generate nine, 9x9 subsamples
subsamp_in = []
subsamp_out = []
for ss in test_input, s in test_input
  xr = Resistance[ss:(ss+Stride-1), s:(s+Stride-1)]
  xo = Origin[ss:(ss+Stride-1), s:(s+Stride-1)]
  X = cat(xr, xo, dims=3)
  Y = Connectivity[ss:(ss+Stride-1), s:(s+Stride-1)]
  push!(subsamp_in, X)
  push!(subsamp_out, Y)
end


#get all cartesian points for range
Px_range = []
for i in 1:length(range)
  u = collect(range[i][1])
  push!(Px_range, u)
end
Px_range = vcat(Px_range...)

Py_range = []
for i in 1:length(range)
  u = collect(range[i][2])
  push!(Py_range, u)
end
Py_range = vcat(Py_range...)


#moving window
#training data within range
X = []
Y = []
for i in Px_range, j in Py_range
  xr = vec(r[i:(i+stride-1),j:(j+stride-1)])
  xo = vec(o[i:(i+stride-1),j:(j+stride-1)])
  x = vcat(xr, xo) #stack the matrices together
  y = c[i:(i+stride-1),j:(j+stride-1)] #matrix we want to predict
  if minimum(y) > 0 #predict only when there is connectivity
    push!(X, x)
    push!(Y, y)
  end
end
X
Y
