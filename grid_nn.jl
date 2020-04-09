using StatsBase
using Flux
using CSV
using Plots
using Random

cd(@__DIR__)

@time begin
  #Read in the CSV (comma separated values) file and convert them to arrays.
  r = CSV.read("data/resistance.csv")
  o = CSV.read("data/origin.csv")
  c = CSV.read("data/connectivity.csv", delim="\t")
  r = convert(Matrix, r)
  o = convert(Matrix, o)
  c = convert(Matrix, c)

  #remove last row in a to get same size as b and c
  r = r[1:end-1, :]

  #change NaN to 0
  function nan_to_0(s)
    for j in 1:length(s)
      if isnan(s[j])
        s[j] = 0
      end
    end
  end

  nan_to_0(r)
  nan_to_0(o)
  nan_to_0(c)
end

stride = 10
grids = 1

Random.seed!(1234)
#select 1 coordinates where there is data
cart_idx = sample(findall(r .> 0), grids)
coordinates = Tuple.(cart_idx)

#testing data
test_input = []
test_output = []
for q in coordinates
  xr = vec(r[first(q):(first(q)+stride-1), last(q):(last(q)+stride-1)])
  xo = vec(o[first(q):(first(q)+stride-1), last(q):(last(q)+stride-1)])
  x_test = vcat(xr, xo)
  y_test = c[first(q):(first(q)+stride-1),last(q):(last(q)+stride-1)]
  push!(test_input, x_test)
  push!(test_output, y_test)
end



#create a range around window
range = []
for f in cart_idx
  a, b = Tuple(f)
  m = [a-10:a+stride+9,b-10:b+stride+9]
  push!(range, m)
end
range

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

train_data = zip(X, Y)

model = Chain(
  Dense(convert(Int,stride*stride)*2,250, σ),
  Dense(250, convert(Int,stride*stride), σ),
  (f) -> reshape(f, (stride,stride))
)

function loss(x, y)
  ŷ = model(x)
  sum((y .- ŷ).^2)./prod(size(x)) #divided by the actual value
end

evalcb() = @show(loss(x_test, y_test))

#train
#for every epoch, sample 500 random maps from the 150 10x10 random matrices/maps
@info("Beginning training loop...")
Random.seed!(1234)
@time @elapsed for epoch in 1:500
  index = sample(1:length(X), 500, replace=false)
  train_data = zip(X[index], Y[index])
  Flux.train!(loss, params(model), train_data, ADAM(0.001), cb = Flux.throttle(evalcb, 1))
end


#have a look
@info "plotting"
p1 = heatmap(test_output[1], title="predicted")
p2 = heatmap(model(test_input[1]), title="observed")
p3 = scatter(test_output[1], model(test_input[1]), leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed", yaxis="predicted")
plot(p1,p2, p3)
