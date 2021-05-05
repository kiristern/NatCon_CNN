@time include("libraries.jl")
@time include("functions.jl") #desired object found in line 23 of preprocess_idx.jl script
@time include("model.jl")


connectivity_renard = readasc("data/maps_for_Kiri/RR_cum_currmap.asc")
resistance_renard = readasc("data/maps_for_Kiri/Resistance_zone_beta_RR.asc"; nd="NODATA")
Origin = readasc("data/input/origin.asc"; nd="NODATA")

begin
    nan_to_0(connectivity_renard)
    nan_to_0(resistance_renard)
    nan_to_0(Origin)
end

#create Training dataset
# Extract 150 random 9x9 resistance, origin, and connectivity layers
Stride = 9
Random.seed!(1234)

maps_fox = []
connect_fox = []
for i in rand(1:size(connectivity_renard,2)-Stride, 300), j in rand(1:size(connectivity_renard,2)-Stride, 300)
  #taking groups of matrices of dimensions StridexStride
  x_res = resistance_renard[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin layers
  y = connectivity_renard[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  # if minimum(y) > 0 #predict only when there is connectivity
    push!(maps_fox, x)
    push!(connect_fox, y)
  # end
end

#create Testing dataset
Random.seed!(5678)

test_maps_fox = []
test_connect_fox = []
for i in rand(1:size(connectivity_renard,2)-Stride, 300), j in rand(1:size(connectivity_renard,2)-Stride, 300)
  #taking groups of matrices of dimensions StridexStride
  x_res = resistance_renard[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin vectors
  y = connectivity_renard[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  # if minimum(y) > 0 #predict only when there is connectivity
    push!(test_maps_fox, x)
    push!(test_connect_fox, y)
  # end
end

#script returns:
maps_fox
connect_fox
test_maps_fox
test_connect_fox

Random.seed!(1234)
train_maps_fox, train_connect_fox, valid_maps_fox, valid_connect_fox = partition_dataset(maps_fox, connect_fox)

batch_size=32
train_set_fox, validation_set_fox = make_sets(train_maps_fox, train_connect_fox, valid_maps_fox, valid_connect_fox)


#TODO: run train_model.jl





p1 = heatmap(validation_set_fox[1][2][:,:,1,32], title="predicted") #connectivity map
p2 = heatmap(model(validation_set_fox[1][1])[:,:,1,32], title="observed") #resistance and origin layer map
p3 = scatter(validation_set_fox[1][2][:,:,1,32], model(validation_set_fox[1][1])[:,:,1,32], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
plot(p1,p2,p3)
# savefig("figures/fox_full_300samples_$(run)sec_$(best_acc*100)%.png")
