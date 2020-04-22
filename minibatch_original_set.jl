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

# include("preprocess.jl")
# include("validation_dataset.jl")

train_maps
train_connect
valid_maps
valid_connect

batch_size = 32 # The CNN only "sees" 32 images at each training cycle

#subtract remainders to ensure all minibatches are the same length
droplast = rem(length(train_maps), batch_size)
mb_idxs = Iterators.partition(1:length(train_maps)-droplast, batch_size)
#train set in the form of batches
train_set = [make_minibatch(train_maps, train_connect, i) for i in mb_idxs]


droplast2 = rem(length(valid_maps), batch_size)
mb_idxs2 = Iterators.partition(1:length(valid_maps)-droplast2, batch_size)
#prepare validation set as one giant minibatch
validation_set = [make_minibatch(valid_maps, valid_connect, i) for i in mb_idxs2]

#Check how data has been arranged
typeof(train_set) #tuple of 4D X_training data and 4D Y_labels
#Check dimensions: width x height x channels x #batches
size(train_set[1][1]) # 9x9x2x32
size(train_set[1][2]) # 9x9x1x32
