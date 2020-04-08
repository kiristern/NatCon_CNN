using CSV
using Flux
using Random

cd(@__DIR__)

#Read in the CSV (comma separated values) file and convert them to arrays.
resistance = CSV.read("data/resistance.csv")
origin = CSV.read("data/origin.csv")
connectivity = CSV.read("data/connectivity.csv", delim="\t")
Resistance = convert(Matrix, resistance)
Origin = convert(Matrix, origin)
Connectivity = convert(Matrix, connectivity)

#remove last row in a to get same size as b and c
Resistance = Resistance[1:end-1, :]

#change NaN to 0
function nan_to_0(s)
  for j in 1:length(s)
    if isnan(s[j])
      s[j] = 0
    end
  end
end

nan_to_0(Resistance)
nan_to_0(Origin)
nan_to_0(Connectivity)

#create Training dataset
# Extract and vectorize 150 random 10x10 resistance, origin, and connectivity layers
Random.seed!(1234)
Stride = 10
train_imgs = []
train_labels = []

for i in rand(10:950, 150), j in rand(10:950, 150)
  #taking groups of matrices of dimensions StridexStride
  x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin vectors
  y = Connectivity[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  if minimum(y) > 0 #predict only when there is connectivity
    push!(imgs, x)
    push!(labels, y)
  end
end

#create Testing dataset
# Extract and vectorize 150 random 10x10 resistance, origin, and connectivity layers
Random.seed!(5678)
Stride = 10
test_imgs = []
test_labels = []

for i in rand(10:950, 150), j in rand(10:950, 150)
  #taking groups of matrices of dimensions StridexStride
  x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin vectors
  y = Connectivity[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  if minimum(y) > 0 #predict only when there is connectivity
    push!(test_imgs, x)
    push!(test_labels, y)
  end
end
