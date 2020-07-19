#=

This script returns a visual representation of the 150 sampling points used for training the model.

Points will update according to the change in RNG/training points.

=#

begin
  bounds_connect = Connectivity[:,1:size(Connectivity,2)-Stride]
  presence_data = findall(x->x > 0, bounds_connect)

  samp_pts = sample(presence_data, 150)

  get_train_samp = []
  get_train_samp2 = []
  for i in 1:length(samp_pts)
    x = samp_pts[i][1]
    y = samp_pts[i][2]
    push!(get_train_samp, y)
    push!(get_train_samp2, x)
  end
end

#visualize samples retrieved for Training
sample_pts = Tuple.(zip(get_train_samp, get_train_samp2))

# heatmap(Origin)
# scatter!(sample_pts, legend=false)

### ploting the boxplots onto map
# heatmap(Origin)
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
  scale(0.40, 0.2905) #scale points to match size of basemap
  Luxor.translate(113.5, 28) #move points to fit within basemap bounds
  setline(1) #width of boxlines
  #get the points used for the training samples
  for i in 1:length(sample_pts)
    rect(sample_pts[i][1], 1255-sample_pts[i][2], 9, 9, :stroke) #create 9x9 rectangles based on the starting points (x,y)
  end
  finish()
  preview()
end
