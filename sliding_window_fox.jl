@time include("libraries.jl")
@time include("functions.jl") #desired object found in line 23 of preprocess_idx.jl script


connectivity_renard = readasc("data/maps_for_Kiri/RR_cum_currmap.asc")
resistance_renard = readasc("data/maps_for_Kiri/Resistance_zone_beta_RR.asc"; nd="NODATA")
Origin = readasc("data/input/origin.asc"; nd="NODATA")

begin
    nan_to_0(connectivity_renard)
    nan_to_0(resistance_renard)
    nan_to_0(Origin)
end

Stride = 9
Desired_x = Int(size(connectivity_renard,2)/9)
rem_desired = rem(size(connectivity_renard, 1), Stride)
Desired_y = Int((size(connectivity_renard, 1)-rem_desired)/9)

c_fox = connectivity_renard[1:end-rem_desired, :]
r_fox = resistance_renard[1:end-rem_desired, :]
o_fox = Origin[1:end-rem_desired, :]

#get coordinates for full connectivity map
all_coord = []
for i in CartesianIndices(c_fox)
  coords = i
  push!(all_coord, coords)
end
all_coord = Tuple.(all_coord)


#create range around first coordinate
first_coor = first(all_coord)
tup1, tup2 = Tuple(first_coor)
range_fox = [tup1:tup1+(size(c_fox,2))-1, tup2:tup2+(size(c_fox,1))-1]

#get every single index in samples
x_idx_fox = collect(range_fox[2])
y_idx_fox = collect(range_fox[1])

#get the first coordinate for each smaller (3x3) sample -- sliding window
x_idxes_slide = x_idx_fox[1:3:end-6]
y_idxes_slide = y_idx_fox[1:3:end-6]

#get the starting coordinates
replicate_x_slide = repeat(x_idxes_slide, inner = 400)
replicate_y_slide = repeat(y_idxes_slide, outer = 415)

#zip coordinates together
zip_slide = Tuple.(zip(replicate_x_slide, replicate_y_slide))
last(zip_slide)


#create Training dataset
# Extract 150 random 9x9 resistance, origin, and connectivity layers
Stride = 9
Random.seed!(1234)

maps_slide = []
connect_slide = []
for i in rand(1:size(connectivity_renard,2)-Stride, 300), j in rand(1:size(connectivity_renard,2)-Stride, 300)
  #taking groups of matrices of dimensions StridexStride
  x_res = resistance_renard[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin layers
  y = connectivity_renard[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  # if minimum(y) > 0 #predict only when there is connectivity
    push!(maps_slide, x)
    push!(connect_slide, y)
  # end
end

#create Testing dataset
Random.seed!(5678)

test_maps_slide = []
test_connect_slide = []
for i in rand(1:size(connectivity_renard,2)-Stride, 300), j in rand(1:size(connectivity_renard,2)-Stride, 300)
  #taking groups of matrices of dimensions StridexStride
  x_res = resistance_renard[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin vectors
  y = connectivity_renard[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  # if minimum(y) > 0 #predict only when there is connectivity
    push!(test_maps_slide, x)
    push!(test_connect_slide, y)
  # end
end

#script returns:
# maps_slide
# connect_slide
# test_maps_slide
# test_connect_slide

Random.seed!(1234)
train_maps_slide, train_connect_slide, valid_maps_slide, valid_connect_slide = partition_dataset(maps_slide, connect_slide)

batch_size=32
train_set_slide, validation_set_slide = make_sets(train_maps_slide, train_connect_slide, valid_maps_slide, valid_connect_slide)

@time include("model.jl")

#TODO: run train_model.jl





p1 = heatmap(validation_set_slide[1][2][:,:,1,9], title="predicted") #connectivity map
p2 = heatmap(model(validation_set_slide[1][1])[:,:,1,9], title="observed") #resistance and origin layer map
p3 = scatter(validation_set_slide[1][2][:,:,1,9], model(validation_set_slide[1][1])[:,:,1,9], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
plot(p1,p2,p3)
# savefig("figures/fox_sliding_window_$(run)sec_$(best_acc*100)%.png")
