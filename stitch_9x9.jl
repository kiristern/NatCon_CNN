#=

Get a 27x27 sample from original data maps and, using the trained (working) model, predict nine 9x9 images and "stitch" together in 3x3 batch to get a final 27x27 image.

Output:
    one 27x27 image; aka, 3 (9x9) by 3 (9x9) images

=#

include("preprocess.jl")
include("validation_dataset.jl")
include("minibatch.jl")
include("train_model.jl")

using StatsBase

maps
connect

#select new coordinates for obtaining n 27x27 samples
stride = 27
samples = 50

Random.seed!(1234)
#select coordinates where there is data
cart_idx = sample(findall(Resistance .> 0), samples)
coordinates = Tuple.(cart_idx)

#obtain 27x27 imgs from sample points
res_input = []
test_output = []
for i in coordinates, j in coordinates
  #taking groups of matrices of dimensions stridexstride
  x_res = Resistance[first(i):(first(i)+stride-1), last(j):(last(j)+stride-1)]
  x_or = Origin[first(i):(first(i)+stride-1), last(j):(last(j)+stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin layers
  y = Connectivity[first(i):(first(i)+stride-1),last(j):(last(j)+stride-1)] #matrix we want to predict
  if minimum(y) > 0 #predict only when there is connectivity
    push!(test_input, x)
    push!(test_output, y)
  end
end

#get indices of each sample point
indices_x = []
for i in test_input

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
