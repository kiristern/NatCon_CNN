#=

Bundle images together with labels and group into minibatches

Input:
    train_maps: n x n x 2
    train_connect: n x n
    valid_maps: n x n x 2
    valid_connect: n x n

Output:
    train_set: n x n x 2 x batch_size
    validation_set: n x n x 1 x batch_size

=#

# include("libraries.jl")
# include("functions.jl")
# include("preprocess.jl")
# include("validation_dataset.jl")

train_maps
train_connect
valid_maps
valid_connect

batch_size = 32 # The CNN only "sees" 32 images at each training cycle

train_set, validation_set = make_sets(train_maps, train_connect, valid_maps, valid_connect)

#Check how data has been arranged
typeof(train_set) #tuple of 4D X_training data and 4D Y_labels
#Check dimensions: width x height x channels x #batches
size(train_set[1][1]) # 9x9x2x32
size(train_set[1][2]) # 9x9x1x32
