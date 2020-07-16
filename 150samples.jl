# visualize points that have been extracted for Training
Random.seed!(1234)
sample_from_x = rand(1:size(Origin,2)-Stride, 150)
sample_from_y = rand(1:size(Origin,2)-Stride, 150)

#visualize samples retrieved for Training
sample_pts = Tuple.(zip(sample_from_x, sample_from_y))
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
