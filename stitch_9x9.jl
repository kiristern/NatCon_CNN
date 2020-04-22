#=

Get a 27x27 sample from original data maps and, using the trained (working) model, predict nine 9x9 images and "stitch" together in 3x3 batch to get a final 27x27 image.

Output:
    one 27x27 image; aka, 3 (9x9) by 3 (9x9) images

=#
cd(@__DIR__)

include("preprocess.jl")
include("validation_dataset.jl")
include("minibatch.jl")
include("train_model.jl")

using StatsBase

#select new coordinates for obtaining n 27x27 samples
stride = 27
samples = 50

Random.seed!(1234)
#select coordinates where there is data
cart_idx = sample(findall(Resistance .> 0), samples)
coordinates = Tuple.(cart_idx)

#create range around each sample point
range = []
for i in cart_idx
  a, b = Tuple(i)
  c = [a-10:a+stride+9,b-10:b+stride+9]
  push!(range, c)
end

#get all indices in range
# indices_x = []
# indices_y = []
# for i in 1:length(range)
#   u = collect(range[i][1])
#   w = collect(range[i][2])
#   push!(indices_x, u)
#   push!(indices_y, w)
# end
# range_idx = zip(indices_x, indices_y)

#make 27x27 layers from coordinates
res_layer = []
ori_layer = []
con_layer = []
for i in coordinates
  x1 = Resistance[first(i):(first(i)+stride-1), last(i):(last(i)+stride-1)]
  x2 = Origin[first(i):(first(i)+stride-1), last(i):(last(i)+stride-1)]
  x3 = Connectivity[first(i):(first(i)+stride-1), last(i):(last(i)+stride-1)]
  push!(res_layer, x1)
  push!(ori_layer, x2)
  push!(con_layer, x3)
end

#make 9x9 layers from coordinates













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
