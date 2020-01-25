import Pkg; Pkg.activate(".")

using Flux
using Plots
using StatsBase
using Random

include("lib.jl")

input = readasc("data/input/resistance.asc"; nd="NODATA")
origin = readasc("data/input/origin.asc"; nd="NODATA")
output = readasc("data/output/connectivity.asc")

# using DelimitedFiles

# convert(Matrix{Float32}, input) |> f -> writedlm("resistance.csv", f)
# convert(Matrix{Float32}, origin) |> f -> writedlm("origin.csv", f)
# convert(Matrix{Float32}, output) |> f -> writedlm("connectivity.csv", f)

#Take 20 000 unique and random cartesian indices (constructing a CartesianIndices from an array makes a range of its indices--ie. points) from the output
Y = unique(rand(CartesianIndices(output), 20_000))
filter!(y -> !isnan(input[y]), Y) #if input is NaN, remove
filter!(y -> 23 <= Tuple(y)[1] <= (size(input, 1)-24) , Y) #filter first element of the tuple between: 23 and size of input-24
filter!(y -> 23 <= Tuple(y)[2] <= (size(input, 2)-24) , Y) #filter second element of the tuple between: 23 and size of input-24

#Create 28x28x1x1 layers
#create a range around (a,b) points of input at position Y
X = Array{Float64,4}[]
for y in Y
    a, b = Tuple(y) #tuple will not change
    m = reshape(input[a-14:a+13,b-14:b+13], (28, 28, 1, 1)) #m is a Y-element 28x28x1x1 (width x height x chanel (ex: greyscale=1, RGB=3) x number/batch) array, where tuples (a,b) of input at position Y, range between a/b-14:a/b+13
    push!(X, m)
end
X

#count how many NaN values in X?
no_nan = filter(i -> !any(isnan.(X[i])), 1:length(X)) #filter values from 1:length(X), !any=*don't stop* if NaN
X = X[no_nan] #remove NaN from X
Y = Y[no_nan] #remove NaN from Y


model = Chain( #Chain=call in sequence
    Conv((3,3), 1=>16, pad=(1,1), relu), #3x3 convolution layer, 1 input channel, 16 output channels, pad="buffer" to prevent loss of info, relu=activation fnc
    MaxPool((2,2)),

    Conv((3,3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    Conv((3,3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10),

    Dense(10, 1, σ)
)

model(X[1])
O = [output[y] for y in Y]

#loss fnc. how bad is model
function loss(x, y)
  ŷ = model(x)
  sum((y .- ŷ).^2)
end

loss(X[1], O[1])

evalcb() = @show(loss(X[1], O[1]))

d = [(X[i], O[i]) for i in 1:length(O)]
@progress "Minibatching" for i in 1:200
    k = sample(d, 100, replace=false) #randomly select 100 values from d, without replacement
    Flux.train!(loss, params(model), k, ADAM(0.001), cb = Flux.throttle(evalcb, 5))
end

loss(X[1], O[1])

# Predictions
pred = fill(NaN, size(output))
relevant_pixels = filter(p -> !isnan(output[p]), CartesianIndices(output))
filter!(p -> Tuple(p)[1] > 18, relevant_pixels)
filter!(p -> Tuple(p)[2] > 18, relevant_pixels)
filter!(p -> Tuple(p)[1] < size(output, 1)-18, relevant_pixels)
filter!(p -> Tuple(p)[2] < size(output, 2)-18, relevant_pixels)

iter = 1
@progress for p in shuffle(relevant_pixels)
    global iter
    i, j = Tuple(p)
    ri = (i-14):(i+13)
    rj = (j-14):(j+13)
    if !any(isnan.(input[ri,rj]))
        pred[p] = model(reshape(input[ri,rj], (28, 28, 1, 1))).data[1]
    end
    iter += 1
    if mod(iter, 100) == 0
        @info iter
        plt = heatmap(pred, clim=(0,1), title="$iter")
        display(plt)
    end
end

heatmap(pred, c=:RdYlGn)
