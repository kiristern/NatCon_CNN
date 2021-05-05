# include("libraries.jl")
# include("functions.jl")
# include("preprocess.jl")
# include("validation_dataset.jl")
# include("minibatch.jl")

# train_set
# validation_set


begin
    print("##########################")
    print("## Constructing model...##")
    print("##########################")
end

m = Chain(
    Conv((3,3), 2=>32, pad=(1,1), relu),
    MaxPool((2,2)),
    Conv((3,3), 32=>64, pad=(1,1), relu),
    MaxPool((2,2)),
    Conv((3,3), 64=>81, pad=(1,1), relu),
    MaxPool((2,2))
)
inputlayersize = Array{Float32}(undef, 9, 9, 2, 32)

model = Chain(
    # Apply a Conv layer to a 2-channel (R & O layer) input using a 3x3 window size, giving a 16-channel output. Output is activated by relu
    Conv((3,3), 2=>32, pad=(1,1), relu),
    # 2x2 window slides over x reducing it to half the size while retaining most important feature information for learning (takes highest/max value)
    MaxPool((2,2)),

    Conv((3,3), 32=>64, pad=(1,1), relu),
    MaxPool((2,2)),

    Conv((3,3), 64=>81, pad=(1,1), relu),
    MaxPool((2,2)),

    # flatten from 3D tensor to a 2D one, suitable for dense layer and training
    flatten,
    Dense(Int(prod(size(m[1:6](inputlayersize)))/batch_size), Stride*Stride),

    #reshape to match Connectivity dimensions
    x -> reshape(x, (Stride, Stride, 1, batch_size))
)



# #View layer outputs
model[1](train_set[1][1]) #layer 1: 9x9x16x32
model[1:2](train_set[1][1]) #layer 2: 4x4x16x32
model[1:3](train_set[1][1]) #layer 3: 4x4x32x32
model[1:4](train_set[1][1]) #layer 4: 2x2x32x32
# reshape layer
model[1:5](train_set[1][1]) #layer 5: 128x32
model[1:6](train_set[1][1]) #layer 6: 81x32
model[1:7](train_set[1][1]) #layer 7: 9x9x1x32
model[1:8](train_set[1][1])
model[1:9](train_set[1][1])
