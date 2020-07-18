# using Images
# using Luxor

# visualize points that have been extracted for Training
Random.seed!(1234)
get_train_samp = rand(1:size(Origin,2)-Stride, 150)
get_train_samp2 = rand(Stride:size(Origin,2)-Stride, 150)

bbox=[]
for i in get_train_samp, j in get_train_samp2
  x_res = Resistance[i:(i+Stride-1),j:(j+Stride-1)]
  push!(bbox, x_res)
end

heatmap(maps[1])

#visualize samples retrieved for Training
sample_pts = Tuple.(zip(get_train_samp, get_train_samp2))

heatmap(Origin)
scatter!(sample_pts, legend=false)


samp_grids = []
for i in all_samp_pts
  y = Connectivity[first(i):first(i)+Stride-1,last(i):last(i)+Stride-1] #matrix we want to predict
    push!(samp_grids, y)
end



#create range around coordinate points
samples_range = []
for i in sample_pts
   a, b = Tuple(i)
   c = [a:a+Stride-1,b:b+Stride-1]
   push!(samples_range, c)
end


#collect x and y values of each 9x9
exes = collect.(first.(samples_range))
wyes = collect.(last.(samples_range))
#create the 9x9
rep_exes = repeat.(exes, outer = 9)
rep_wyes = repeat.(wyes, inner = 9)
#zip them together
zip_reps = Tuple.(zip(rep_exes, rep_wyes))
samp_coords = [Tuple.(CartesianIndex.(zip_reps[i]...)) for i in 1:length(zip_reps)]
all_samp_pts = reduce(vcat, samp_coords)

all_samp_pts[1]
Origin[all_samp_pts





### drawing just the points ###
@png begin
  Drawing(1206, 1255, "boxplot_train_samp.png")
  # Luxor.scale(-1)
  # Luxor.scale(0,-1)
  sethue("red")
  setline(2)
  for i in 1:length(sample_pts)
    rect(sample_pts[i][1], 1255-sample_pts[i][2], 9, 9, :stroke)
  end
end





### ploting the boxplots onto map
heatmap(Origin)
# savefig("Origin.png")

begin
  O_img = readpng("Origin.png")
  w = O_img.width
  h = O_img.height
  #create a drawing surface of the same size
  fname = "boxplotsamples_on_map.png"
  Drawing(w, h, fname)
  #place the image on the Drawing -- it's positioned by its top/left corner
  placeimage(O_img, 0,0)
  # now annotate the image. The (0/0) is at the top left.
  sethue("red")
  scale(0.28)
  Luxor.translate(169.5, 45.5)
  # rect(0,0, 9,9, :stroke)
  setline(1)
  # scale(0.30)
  for i in 1:length(sample_pts)
    rect(sample_pts[i][1], 1255-sample_pts[i][2], 9, 9, :stroke)
  end
  finish()
  preview()
end

heatmap(Origin)
scatter!(sample_pts, legend=false)



#with transparent background
Drawing(1255, 1206, "boxplot_train_samp.png")
background(1, 1, 1, 0)
setline(30)
setcolor("red") # current opacity is now 0.0, so use setcolor() rather than sethue()
rect(60,1400, 9,9, :stroke)
finish()
preview()
