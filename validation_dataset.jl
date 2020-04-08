##CREATE A CUSTOM VALIDATION SET BY SAMPLING FROM THE TRAINING DATASET.

include("preprocess.jl")

imgs
labels

function partition_dataset(imgs, labels, valid_ratio=0.1, Shuffle=true)
  """
  Args:
  imgs: array representing the image set from which the partitioning is made.
  labels: the labels associated with the provided images.
  valid_ratio (optional): the portion of the data that will be used in the validation set. Default: 0.1.
  shuffle (optional): whether or not to shuffle the data. Default: True.

  Return:
  A tuple of 4 elements (train_imgs, train_labels, valid_imgs, valid_labels) where:
  train_imgs: an array of images for the training set.
  train_labels: labels associated with the images in the training set.
  valid_imgs: an array of images for the validation set.
  valid_labels: labels associated with the images in the validation set.
  """
  if Shuffle == true
    indices = shuffle(collect(1:size(imgs,1)))
  else
    indices = collect(1:size(imgs,1))
  end

  n_training = Int(round((1.0 - valid_ratio)*length(indices)))
  train_idx, valid_idx = indices[1:n_training], indices[n_training+1:end]

  train_imgs, valid_imgs = imgs[train_idx], imgs[valid_idx]
  train_labels, valid_labels = labels[train_idx], labels[valid_idx]
  return train_imgs, train_labels, valid_imgs, valid_labels
end

Random.seed!(1234)
train_imgs, train_labels, valid_imgs, valid_labels = partition_dataset(imgs, labels)
