#=
Transform, shuffle and batch datasets
=#

using Flux.Data
using Printf
include("validation_dataset.jl")

train_loader.data # original dataset

batch_size=32
epochs = 5

train_loader = DataLoader(train_maps, train_connect, batchsize=batch_size, partial=false, shuffle=true)
    #If shuffle=true, shuffles the observations each time iterations are re-started.
    #If partial=false, drops the last mini-batch if it is smaller than the batchsize.
for epoch in 1:epochs
    print("Epoch $epoch/$epochs")
    #At every iteration, the dataloader returns a mini-batch of batch_size input-label pairs (x, y).
    for i, (x, y) in train_loader
        print("    batch $i/$(@sprintf("%.0f", ceil(n/batch_size))) of $(size(y,1)) examples")
    end
end




#Flux reference
Xtrain = rand(10, 100)
Ytrain = rand(100)
train_loader = DataLoader(Xtrain, Ytrain, batchsize=2, shuffle=true)
for epoch in 1:100
    for (x, y) in train_loader
        @assert size(x) == (10, 2)
        @assert size(y) == (2,)
        ...
    end
end



#python reference
n = 100
batch_size = 32
train_dataset = create_dataset(train_imgs, train_labels, n)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)

epochs = 2
for epoch in range(epochs):
  print('Epoch {}/{}:'.format(epoch+1, epochs))
  #At every iteration, the dataloader returns a mini-batch of batch_size input-label pairs (x, y).
  for i, (x, y) in enumerate(train_dataloader):
    print('   batch {}/{} of {} examples.'.format(i+1, int(np.ceil(n/batch_size)), y.size(0)))
        #<np.ceil> smallest integer i, such that i >= x ie. rounded whole value
