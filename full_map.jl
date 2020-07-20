begin
  Stride = 9
  Desired_x = Int(size(Connectivity,2)/9)
  rem_desired = rem(size(Connectivity, 1), Stride)
  Desired_y = Int((size(Connectivity, 1)-rem_desired)/9)

  c = Connectivity[1:end-rem_desired, :]
  r = Resistance[1:end-rem_desired, :]
  o = Origin[1:end-rem_desired, :]

  #get coordinates for full connectivity map
  all_coord = []
  for i in CartesianIndices(c)
    coords = i
    push!(all_coord, coords)
  end
  all_coord = Tuple.(all_coord)


  #create range around first coordinate
  first_coor = first(all_coord)
  tup1, tup2 = Tuple(first_coor)
  range = [tup1:tup1+(size(c,2))-1, tup2:tup2+(size(c,1))-1]

  #get every single index in samples
  x_idx = collect(range[2])
  y_idx = collect(range[1])

  #get the first coordinate for each smaller (9x9) sample
  x_idxes = x_idx[1:Stride:end]
  y_idxes = y_idx[1:Stride:end]

  #get the 9 starting coordinates
  replicate_x = repeat(x_idxes, inner = length(y_idxes))
  replicate_y = repeat(y_idxes, outer = length(x_idxes))

  #zip coordinates together
  zip_coor = Tuple.(zip(replicate_x, replicate_y))
  last(zip_coor)

  #create 9x9 samples
  maps9x9 = []
  connect9x9 = []
  for (i,j) in zip_coor
    x_res = r[i:(i+Stride-1),j:(j+Stride-1)]
    x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
    x = cat(x_res, x_or, dims=3)
    y = c[i:(i+Stride-1),j:(j+Stride-1)]
    push!(maps9x9, x)
    push!(connect9x9, y)
  end


  batch_size=32
  ### minibatch ###
  #subtract remainders to ensure all minibatches are the same length
  droplast9x9 = rem(length(maps9x9), batch_size)
  mb_idxs9x9 = Iterators.partition(1:length(maps9x9)-droplast9x9, batch_size)
  #train set in the form of batches
  nine_nine = [make_minibatch(maps9x9, connect9x9, i) for i in mb_idxs9x9]



  ### verify connectivity values are the same ###
  truemap = [reduce(hcat, p) for p in Iterators.partition(connect9x9, Desired_x)]
  truemap = [reduce(vcat, p) for p in Iterators.partition(truemap, Desired_y)]
  # heatmap(truemap_fox[1])
  all(isapprox.(c, truemap[1]))
end


#load saved model
@time include("model.jl")
@time @load "BSON/bear3rdmod_no0s.bson" params #upload last saved model
Flux.loadparams!(model, params) #new model will now be identical to the one saved params for

##### Run saved model on data #####
begin
  model_on_9x9 = trained_model(nine_nine)


  #reduce 4D to 2D
  begin
    mod = []
    for t in model_on_9x9
      tmp2 = [t[:,:,1,i] for i in 1:batch_size]
      push!(mod, tmp2)
    end
    #reduce to one vector of arrays
    mod = reduce(vcat, mod)

    # remove_last = rem(length(mod), 9)
    #hcat groups of three
    stitched = [reduce(hcat, p) for p in Iterators.partition(mod, Desired_x)]
    #vcat the stitched hcats
    stitchedmap = [reduce(vcat, p) for p in Iterators.partition(stitched[1:end-1], length(stitched))]
  end
end

minimum(stitchedmap[1])
maximum(stitchedmap[1])
count(x->x > 1, stitchedmap[1])
count(x->x <0, stitchedmap[1])
count(x->x ==0, stitchedmap[1])
# findall(x->x >1, stitchedmap[1])
# replace!(stitchedmap[1], 0 => NaN)




# originalmap = heatmap(c)
# savefig("figures/fox.png")

fullmap = heatmap(stitchedmap[1])
savefig("figures/bearmod10_sampleonlywheredata_onbear.png")

scat1 = scatter(stitchedmap[1], c[1:end-9, :], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
savefig("figures/scatter_bearmod10_sampleonlywheredata_on_bear.png")

difference1 = stitchedmap[1] - c[1:end-9, :] #overestimating = 1; underestimating = -1
heatmap(difference1)
savefig("figures/difference_bearmod10_sampleonlywheredata_on_bear.png")



### IF < 0, set to 0 ###
begin
  #if less than 0, = 0; if >1 = 1
  model_on_9x9_zero = replace.(x -> x < 0 ? 0 : x, model_on_9x9)
  # model_9x9 = replace.(x -> x > 1 ? 1 : x, model_on_9x9_zero)

  #reduce 4D to 2D
  mod0 = []
  for t in model_on_9x9_zero
    tmp2 = [t[:,:,1,i] for i in 1:batch_size]
    push!(mod0, tmp2)
  end
  mod0 = reduce(vcat, mod0)

  stitched0 = [reduce(hcat, p) for p in Iterators.partition(mod0, Desired_x)]
  stitchedmap0 = [reduce(vcat, p) for p in Iterators.partition(stitched0[1:end-1], length(stitched0))]
end

heatmap(stitchedmap0[1])
savefig("figures/bearmod10_sampleonlywheredata_on_bear<0.png")





#normalize data between 0-1 while keeping weights
map_scale0 = (stitchedmap0[1] .- minimum(stitchedmap0[1])) ./ (maximum(stitchedmap0[1]) .- minimum(stitchedmap0[1]))

heatmap(map_scale0)
savefig("figures/bearmod10_sampleonlywheredata_on_bear<0scaled.png")

scat2 = scatter(map_scale0, c[1:end-9, :], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
savefig("figures/scatter_bearmod10_sampleonlywheredata_on_bear<0scaled.png")

dif2 = map_scale0 - c[1:end-9, :]
heatmap(dif2)
savefig("figures/difference_bearmod10_sampleonlywheredata_on_bear<0scaled.png")






#### 0 =< x <= 1 ####
begin
  model_9x9 = replace.(x -> x > 1 ? 1 : x, model_on_9x9_zero)

  #reduce 4D to 2D
  mod2 = []
  for t in model_9x9
    tmp2 = [t[:,:,1,i] for i in 1:batch_size]
    push!(mod2, tmp2)
  end
  mod2 = reduce(vcat, mod2)

  stitched1 = [reduce(hcat, p) for p in Iterators.partition(mod2, Desired_x)]
  stitchedmap1 = [reduce(vcat, p) for p in Iterators.partition(stitched1[1:end-1], length(stitched1))]
end

heatmap(stitchedmap1[1])
savefig("figures/bearmod9layers_sampleonlywheredata_on_bear0<<1.png")

scat3 = scatter(stitchedmap1[1], c[1:end-9, :], leg=false, c=:black, xlim=(0,1), ylim=(0,1), xaxis="observed (model)", yaxis="predicted (true values)")
savefig("figures/scatter_bearmod9layers_sampleonlywheredata_on_bear0<<1.png")

dif3 = stitchedmap1[1] - c[1:end-9, :]
heatmap(dif3)
savefig("figures/difference_bearmod9layers_sampleonlywheredata_on_bear0<<1.png")
