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
window = 2

Random.seed!(1234)
#select coordinates where there is data
Q = sample(findall(r .> 0), window)
Qprime = Tuple.(Q)


W = []
Z = []
for q in Qprime
  xr = vec(r[first(q):(first(q)+stride-1), last(q):(last(q)+stride-1)])
  xo = vec(o[first(q):(first(q)+stride-1), last(q):(last(q)+stride-1)])
  x_test = vcat(xr, xo)
  y_test = c[first(q):(first(q)+stride-1),last(q):(last(q)+stride-1)]
  push!(W, x_test)
  push!(Z, y_test)
end
W
Z


#create a range around window
range = []
for f in Q
  a, b = Tuple(f)
  m = [a-10:a+stride+9,b-10:b+stride+9]
  push!(range, m)
end
range

P1 = []
for i in 1:length(range)
  u = collect(range[i][1])
  push!(P1, u)
end
P1 = vcat(P1...)

P2 = []
for i in 1:length(range)
  u = collect(range[i][1])
  push!(P2, u)
end
P2 = vcat(P2...)


X = []
Y = []
for i in P1, j in P2
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
  Dense(convert(Int,stride*stride)*2,100, σ),
  Dense(100, convert(Int,stride*stride), σ),
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
p1 = heatmap(y_test, title="predicted")
p2 = heatmap(model(x_test, title="observed")
p3 = scatter(y_test, model(x_test), leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed", yaxis="predicted")
plot(p1,p2, p3)
