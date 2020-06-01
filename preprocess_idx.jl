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

#create range around each sample index (length of stride)
range = []
for i in cart_idx
  a, b = Tuple(i)
  c = [a:a+stride-1,b:b+stride-1]
  push!(range, c)
end

#make 27x27 imgs from coordinates
validate_map27x27 = []
validate_connect27x27 = []
for i in coordinates
  x_resistance = Resistance[first(i):first(i)+stride-1,last(i):last(i)+stride-1]
  x_origin = Origin[first(i):first(i)+stride-1,last(i):last(i)+stride-1]
  x = cat(x_resistance, x_origin, dims=3) #concatenate resistance and origin layers
  y = Connectivity[first(i):first(i)+stride-1,last(i):last(i)+stride-1] #matrix we want to predict
  if minimum(y) > 0 #predict only when there is connectivity
    push!(validate_map27x27, x)
    push!(validate_connect27x27, y)
  end
end
validate_map27x27
validate_connect27x27

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

#get starting coordinates, (a,b) (a,b+9), (a,b+18)
#replicate the first element in the cartesian tuple 3 times
replicate = first.(repeat(coordinates, inner=3))

#zip coordinates together
dup_coor = []
for i in 1:length(replicate)
  zip_dup = Tuple.(zip(replicate[i], y_idxes[i])) #y_idxes from preprocess_idx.jl script
  push!(dup_coor, zip_dup)
end
dup_coor

#create 9x9 samples
maps9x9 = []
connect9x9 = []
for i in first.(dup_coor), j in last.(dup_coor)
  x_res2 = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  x_or2 = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x2 = cat(x_res2, x_or2, dims=3) #concatenate resistance and origin layers
  y2 = Connectivity[i:(i+desired-1),j:(j+desired-1)] #matrix we want to predict
  if minimum(y2) > 0 #predict only when there is connectivity
    push!(maps9x9, x2)
    push!(connect9x9, y2)
  end
end

maps9x9
connect9x9

validate_connect27x27
heatmap(validate_connect27x27[1])

### minibatch ###
#subtract remainders to ensure all minibatches are the same length
droplast9x9 = rem(length(maps9x9), batch_size)
mb_idxs9x9 = Iterators.partition(1:length(maps9x9)-droplast9x9, batch_size)
#train set in the form of batches
nine_nine = [make_minibatch(maps9x9, connect9x9, i) for i in mb_idxs]
nine_nine



#TODO: verify connectivity values are the same
truem1_2 = connect9x9[1]
truem2_2 = connect9x9[2]
truem3_2 = connect9x9[3]
truerow1_2 = hcat(truem1_2,truem2_2,truem3_2)

truem4_2 = connect9x9[4]
truem5_2 = connect9x9[5]
truem6_2 = connect9x9[6]
truerow2_2 = hcat(truem4_2,truem5_2,truem6_2)

truem7_2 = connect9x9[7]
truem8_2 = connect9x9[8]
truem9_2 = connect9x9[9]
truerow3_2 = hcat(truem7_2,truem8_2,truem9_2)

truemap27x27_2 = vcat(truerow1_2, truerow2_2, truerow3_2)
truemap1_2 = heatmap(truemap27x27_2)

plot(heatmap(validate_connect27x27[1]), truemap1_2)

#compare minibatching
truem1_3 = nine_nine[1][2][:,:,1,1]
truem2_3 = nine_nine[1][2][:,:,1,2]
truem3_3 = nine_nine[1][2][:,:,1,3]
truerow1_3 = hcat(truem1_3,truem2_3,truem3_3)

truem4_3 = nine_nine[1][2][:,:,1,4]
truem5_3 = nine_nine[1][2][:,:,1,5]
truem6_3 = nine_nine[1][2][:,:,1,6]
truerow2_3 = hcat(truem4_3,truem5_3,truem6_3)

truem7_3 = nine_nine[1][2][:,:,1,7]
truem8_3 = nine_nine[1][2][:,:,1,8]
truem9_3 = nine_nine[1][2][:,:,1,9]
truerow3_3 = hcat(truem7_3,truem8_3,truem9_3)

truemap27x27_3 = vcat(truerow1_3, truerow2_3, truerow3_3)
truemap1_3 = heatmap(truemap27x27_3)

plot(truemap1_2, truemap1_3, heatmap(validate_connect27x27[1])) #minibatching does not have an effect
