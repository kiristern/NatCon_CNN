using CSV
using Flux #v0.10.4
using Random

cd(@__DIR__)

#Read in the CSV (comma separated values) file and convert them to arrays.
resistance = CSV.read("data/resistance.csv")
origin = CSV.read("data/origin.csv")
connectivity = CSV.read("data/connectivity.csv", delim="\t")
Resistance = convert(Matrix, resistance)
Origin = convert(Matrix, origin)
Connectivity = convert(Matrix, connectivity)

#remove last row in Resistance to get same size as Origin and Connectivity
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
# Extract 150 random 28x28 resistance, origin, and connectivity layers
Random.seed!(1234)
Stride = 6
maps = []
connect = []

for i in rand(10:950, 150), j in rand(10:950, 150)
  #taking groups of matrices of dimensions StridexStride
  x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin layers
  y = Connectivity[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  if minimum(y) > 0 #predict only when there is connectivity
    push!(maps, x)
    push!(connect, y)
  end
end

#create Testing dataset
Random.seed!(5678)
Stride = 6
test_maps = []
test_connect = []

for i in rand(10:950, 150), j in rand(10:950, 150)
  #taking groups of matrices of dimensions StridexStride
  x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin vectors
  y = Connectivity[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  if minimum(y) > 0 #predict only when there is connectivity
    push!(test_maps, x)
    push!(test_connect, y)
  end
end

#script returns:
maps
connect
test_maps
test_connect
