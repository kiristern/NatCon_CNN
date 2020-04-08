## TRANSFORM, SHUFFLE, AND BATCH DATASETS

using Flux.Data

include("validation_dataset.jl")

batch_size=32
epochs = 5

train_loader = DataLoader(train_imgs, train_labels, batchsize=batch_size, shuffle=true)
for epoch in 1:epochs
    for (x, y) in train_loader
        @assert size(x) == (10, 2)
        @assert size(y) == (2,)
        ...
    end
end

# train for 5 epochs
using IterTools: ncycle
Flux.train!(loss, ps, ncycle(train_loader, 5), opt)
