include("preprocess.jl")

imgs
labels



function partition_dataset(imgs, labels, valid_ratio=0.1, Shuffle=true, seed=1234)
  """
  Args:
  imgs: array representing the image set from which the partitioning is made.
  labels: the labels associated with the provided images.
  valid_ratio (optional): the portion of the data that will be used in the validation set. Default: 0.1.
  shuffle (optional): whether or not to shuffle the data. Default: True.
  seed (optional): the seed of the numpy random generator: Default: 1234.

  Return:
  A tuple of 4 elements (train_imgs, train_labels, valid_imgs, valid_labels) where:
  train_imgs: an array of images for the training set.
  train_labels: labels associated with the images in the training set.
  valid_imgs: an array of images for the validation set.
  valid_labels: labels associated with the images in the validation set.
  """
  if Shuffle == true
    # Random.seed!(seed)
    #rng = MersenneTwister(seed)
    indices = shuffle(1:size(imgs,1))
  else
    indices = collect(1:size(imgs,1))
  end

  #train, test = partition(eachindex(y), 0.7, shuffle=true)
  n_training = Int(round((1.0 - valid_ratio)*length(indices)))
  train_idx, valid_idx = indices[1:n_training], indices[n_training+1:end]
  #=
  train_idx, valid_idx = np.split(
  indices,
  [int((1.0 - valid_ratio)*len(indices))]
  )
  =#
  train_imgs, valid_imgs = imgs[train_idx], imgs[valid_idx]
  #tgt = np.array(labels)
  #train_labels, valid_labels = tgt[train_idx].tolist(), tgt[valid_idx].tolist()
  train_labels, valid_labels = labels[train_idx], labels[valid_idx]
  return train_imgs, train_labels, valid_imgs, valid_labels
end

Random.seed!(1234)
partition_dataset(imgs, labels)



def partition_dataset(imgs, labels, valid_ratio=0.1, shuffle=True, seed=1234):
    """
    Args:
       imgs: numpy array representing the image set from which
          the partitioning is made.
       labels: the labels associated with the provided images.
       valid_ratio (optional): the portion of the data that will be used in
          the validation set. Default: 0.1.
       shuffle (optional): whether or not to shuffle the data. Default: True.
       seed (optional): the seed of the numpy random generator: Default: 1234.

    Return:
       A tuple of 4 elements (train_imgs, train_labels, valid_imgs, valid_labels)
       where:
          train_imgs: a numpy array of images for the training set.
          train_labels: labels associated with the images in the training set.
          valid_imgs: a numpy array of images for the validation set.
          valid_labels: labels associated with the images in the validation set.

    """
    if shuffle:
      np.random.seed(seed)  # Set the random seed of numpy.
      indices = np.random.permutation(imgs.shape[0])
        #<np.random.permutation> Randomly permute a sequence. If x is a multi-dimensional array, it is only shuffled along its first index.
        #<.shape> returns dimensions of the array
          #if Y has n rows and m cols, Y.shape is (n,m)
        #<imgs.shape[0]> is n of imgs
    else:
      indices = np.arange(imgs.shape[0])
      #<np.arange> Return evenly spaced values within a given interval

    train_idx, valid_idx = np.split( #split indices into train and valid index based on valid_ratio
        indices,
        [int((1.0 - valid_ratio)*len(indices))]
    )
    train_imgs, valid_imgs = imgs[train_idx], imgs[valid_idx]
    tgt = np.array(labels)
    train_labels, valid_labels = tgt[train_idx].tolist(), tgt[valid_idx].tolist()
    return train_imgs, train_labels, valid_imgs, valid_labels
