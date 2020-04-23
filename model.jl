include("libraries.jl")
include("functions.jl")
include("preprocess.jl")
include("validation_dataset.jl")
include("minibatch.jl")

####
# First run
####

train_set
validation_set

begin
    print("##########################")
    print("## Constructing model...##")
    print("##########################")
end

#Needed to update model
m = Chain(
    Conv((3,3), 2=>16, pad=(1,1), relu),
    MaxPool((2,2)),
    Conv((3,3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2))
)

ls = m[1:4](train_set[1][1])
reshapeLayer = size(ls,1)*size(ls,2)*size(ls,3)
model = Chain(
    #Apply a Conv layer to a 2-channel input using a 2x2 window size, giving a 16-channel output. Output is activated by relu
    Conv((3,3), 2=>16, pad=(1,1), relu),
    MaxPool((2,2)),
    #2x2 window slides over x reducing it to half the size while retaining most important feature information for learning (takes highest/max value)
    Conv((3,3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    #flatten from 3D tensor to a 2D one, suitable for dense layer and training
    x -> reshape(x, (reshapeLayer, batch_size)),

     Dense(reshapeLayer, Stride*Stride),

    #reshape to match output dimensions
    x -> reshape(x, (Stride, Stride, 1, batch_size))
)

#View layer outputs
model[1](train_set[1][1]) #layer 1: 9x9x16x32
model[1:2](train_set[1][1]) #layer 2: 4x4x16x32
model[1:3](train_set[1][1]) #layer 3: 4x4x32x32
model[1:4](train_set[1][1]) #layer 4: 2x2x32x32
model[1:5](train_set[1][1]) #layer 5: 128x32
model[1:6](train_set[1][1]) #layer 6: 81x32
model[1:7](train_set[1][1]) #layer 7: 9x9x1x32


####
# Second run
####

train_set2
validation_set2

#Needed to update model
m = Chain(
    Conv((3,3), 2=>16, pad=(1,1), relu),
    MaxPool((2,2)),
    Conv((3,3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2))
)

ls2 = m[1:4](train_set2[1][1])
reshapeLayer2 = size(ls2,1)*size(ls2,2)*size(ls2,3)

begin
    print("##########################")
    print("## Constructing model 2...##")
    print("##########################")
end
model = Chain(
    #Apply a Conv layer to a 2-channel input using a 2x2 window size, giving a 16-channel output. Output is activated by relu
    Conv((3,3), 2=>16, pad=(1,1), relu),
    MaxPool((2,2)),
    #2x2 window slides over x reducing it to half the size while retaining most important feature information for learning (takes highest/max value)
    Conv((3,3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    #flatten from 3D tensor to a 2D one, suitable for dense layer and training
    x -> reshape(x, (reshapeLayer2, batch_size2)),

     Dense(reshapeLayer2, Stride*Stride),

    #reshape to match output dimensions
    x -> reshape(x, (Stride, Stride, 1, batch_size2))
)

#View layer outputs
model[1](train_set2[1][1]) #layer 1: 9x9x16x32
model[1:2](train_set2[1][1]) #layer 2: 4x4x16x32
model[1:3](train_set2[1][1]) #layer 3: 4x4x32x32
model[1:4](train_set2[1][1]) #layer 4: 2x2x32x32
model[1:5](train_set2[1][1]) #layer 5: 128x32
model[1:6](train_set2[1][1]) #layer 6: 81x32
model[1:7](train_set2[1][1]) #layer 7: 9x9x1x32
