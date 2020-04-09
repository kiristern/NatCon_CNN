#=
Create a custom validation set by sampling from the training dataset
=#

include("preprocess.jl")

maps
connect

function partition_dataset(maps, connect, valid_ratio=0.1, Shuffle=true)
  """
  Args:
  maps: array representing the image set from which the partitioning is made.
  connect: the connect associated with the provided images.
  valid_ratio (optional): the portion of the data that will be used in the validation set. Default: 0.1.
  shuffle (optional): whether or not to shuffle the data. Default: True.

  Return:
  A tuple of 4 elements (train_maps, train_connect, valid_maps, valid_connect) where:
  train_maps: an array of images for the training set.
  train_connect: connect associated with the images in the training set.
  valid_maps: an array of images for the validation set.
  valid_connect: connect associated with the images in the validation set.
  """
  if Shuffle == true
    indices = shuffle(collect(1:size(maps,1)))
  else
    indices = collect(1:size(maps,1))
  end

  n_training = Int(round((1.0 - valid_ratio)*length(indices)))
  train_idx, valid_idx = indices[1:n_training], indices[n_training+1:end]

  train_maps, valid_maps = maps[train_idx], maps[valid_idx]
  train_connect, valid_connect = connect[train_idx], connect[valid_idx]
  return train_maps, train_connect, valid_maps, valid_connect
end

Random.seed!(1234)
train_maps, train_connect, valid_maps, valid_connect = partition_dataset(maps, connect)
