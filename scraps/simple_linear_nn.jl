using Flux
using StatsBase
using CSV
using Printf

cd(@__DIR__)

@time begin
  #Read in the CSV (comma separated values) file and convert them to arrays.
  a = CSV.read("../data/resistance.csv")
  b = CSV.read("../data/origin.csv")
  c = CSV.read("../data/connectivity.csv", delim="\t")
  a = convert(Matrix, a)
  b = convert(Matrix, b)
  c = convert(Matrix, c)

  #remove last row in a to get same size as b and c
  a = a[1:end-1, :]

  #change NaN to 0
  R = a
  for j in 1:length(a)
    if isnan(a[j])
      R[j] = 0
    end
  end

  O = b
  for j in 1:length(b)
    if isnan(b[j])
      O[j] = 0
    end
  end

  C = c
  for j in 1:length(c)
    if isnan(c[j])
      C[j] = 0
    end
  end

  R = vec(R)
  O = vec(O)
  C = vec(C)

  df = [R O C]

  #Split our dataset into the input features (which we call x) and the label (which we call y).
  #need to take transpose or it doesn't work...??
  x = df[:, 1:2]'
  y = df[:, 3:3]'

  #Split our dataset into the training set, the validation set and the test set.
  split_ratio = 0.1 # For the train test split

  # Split into train and test sets
  split_index = floor(Int,size(x,2)*split_ratio)
  x_train = x[:,1:split_index]
  y_train = y[:,1:split_index]
  x_test = x[:,split_index+1:size(x,2)]
  y_test = y[:,split_index+1:size(x,2)]

  # train_data = [(x_train, y_train)]
  # train_data = Iterators.repeated((x_train, y_train), 3) #train model on the same data 3 times
  train_data = zip(x_train, y_train)
  test_data = zip(x_test, y_test)
end

#create model
#params() to keep track of these values
W = param(randn(1,2)/10) #assign random weight
b = param([0.]) #add bias... 0 bias?

#simple linear regression to predict an output array y from input x
model = predict(x) = W*x .+ b

function loss(x, y)
  ŷ = predict(x)
  sum((y .- ŷ).^2)
end

#verify loss function
findall(x->x==1.0, x) #search for x and y value  that's not zero (ie. 1.0) to test loss function
loss(x[726], y[726])

#to improve prediction, take gradients of W and b wrt loss and perform gradient descent
gs = gradient(() -> loss(x_train,y_train), params([W,b]))
#pull out gradients to update W to train the model
W̄ = gs[W]
W .-= 0.1 .* W̄
loss(x[726], y[726])

opt = ADAM(0.001) #learn rate (η = 0.01)

#evaluate callback (ie. to observe the training process)
evalcb() = @show(loss(x_test, y_test))

@info("Beginning training loop...")
Flux.train!(loss, params(model), train_data, opt, cb = Flux.throttle(evalcb, 5))

#train for 2 epochs (ie. how many times train! loops over data)
@Flux.epochs 2 Flux.train!(loss, params(model), train_data, opt, cb = Flux.throttle(evalcb, 5))

#MSE/loss
meansquarederror(ŷ, y) = sum((ŷ .- y).^2)/size(y, 2)
err = meansquarederror(predict(x_test),y_test)
println(err)


function update!(x::AbstractArray, x̄)
  x .+= x̄
  return x
end

η = 0.001

for i = 1:100
  gs = gradient(() -> loss(x_train,y_train), params([W,b]))
  for x in params([W, b])
    update!(x, -gs[x]*η)
  end
  @show loss(x_train, y_train)
end
