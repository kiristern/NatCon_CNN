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
# @time include("libraries.jl")
# @time include("functions.jl")
# @time include("preprocess.jl")

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

#create range around each sample index
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
validation_maps
validation_connect

#get every single index in samples
x_indices = []
y_indices = []
for i in 1:length(range)
  x_idx = collect(range[i][1])
  y_idx = collect(range[i][2])
  push!(x_indices, x_idx...)
  push!(y_indices, y_idx...)
end
x_indices
y_indices


#get the first coordinate for each smaller (9x9) sample
x_idxes = x_indices[1:Stride:end]
y_idxes = y_indices[1:Stride:end]

#create 9x9 samples
maps9x9 = []
connect9x9 = []
for i in x_idxes, j in y_idxes
  x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin layers
  y = Connectivity[i:(i+desired-1),j:(j+desired-1)] #matrix we want to predict
  if minimum(y) > 0 #predict only when there is connectivity
    push!(maps9x9, x)
    push!(connect9x9, y)
  end
end

maps9x9
connect9x9
validation_maps
validation_connect
