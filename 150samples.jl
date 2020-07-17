# using Images
# using Luxor

# visualize points that have been extracted for Training
Random.seed!(1234)
get_train_samp = rand(Stride:size(Origin,2)-Stride, 150)
get_train_samp2 = rand(Stride:size(Origin,2)-Stride, 150)


maps = []
connect = []
for i in get_train_samp, j in get_train_samp2
  x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  x_or = Origin[i:(i+Stride-1),j:(j+Stride-1)]
  x = cat(x_res, x_or, dims=3) #concatenate resistance and origin layers
  y = Connectivity[i:(i+Stride-1),j:(j+Stride-1)] #matrix we want to predict
  push!(maps, x)
  push!(connect, y)
end


#visualize samples retrieved for Training
sample_pts = Tuple.(zip(get_train_samp, get_train_samp2))

heatmap(Origin)
scatter!(sample_pts, legend=false)

#create range around coordinate points
sample_range = []
for i in sample_pts
   a, b = Tuple(i)
   c = [a:a+Stride-1,b:b+Stride-1]
   push!(sample_range, c)
end

#collect x and y values of each 9x9
exes = collect.(first.(sample_range))
wyes = collect.(last.(sample_range))
#create the 9x9
rep_exes = repeat.(exes, outer = 9)
rep_wyes = repeat.(wyes, inner = 9)
#zip them together
zip_reps = Tuple.(zip(rep_exes, rep_wyes))
samp_coords = [Tuple.(CartesianIndex.(zip_reps[i]...)) for i in 1:length(zip_reps)]
all_samp_pts = reduce(vcat, samp_coords)

display(plot(samp_coords))
