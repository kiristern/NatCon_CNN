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

#extract and vectorize values for 10x10 resistance, origin and connectivity layers
xr = vec(r[200:209,200:209]) #resistance
xo = vec(o[200:209,200:209]) #origin
x_test = vcat(xr,xo)
y_test = c[200:209,200:209] #connectivity - what we want to predict




Random.seed!(1234)
X = []
Y = []
#taking 150 random 10x10 matrices of r, o and c layers
for i in rand(10:950, 150), j in rand(10:950, 150)
  #taking groups of matrices of dimensions stridexstride
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
