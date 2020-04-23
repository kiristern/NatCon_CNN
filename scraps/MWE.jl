#ex_input = [(Array{Float32}(undef, 3, 3, 2)), (Array{Float32}(undef, 3, 3, 2))]
#input = fill([ex_input], 5)
ex_output = Array{Float32}(undef, 3, 3)
output = fill(ex_output, 5)
ex_input = Array{Float32}(undef, 3, 3, 2)
input = fill(ex_input, 5)


randn(64,64,1,1) |>
    Conv((3,3), 1=>8, relu, stride=2, pad=1) #|>
    #ConvTranspose((3,3), 8=>1, relu, stride=2, pad=1) #|> size
# Due to striding, losing pixels on the borders (the filters can't reach them) on the input -> returns (63, 63, 1, 1) instead of (64,64,1,1)

#Solve with unequal padding
randn(64,64,1,1) |>
    Conv((3,3), 1=>8, leakyrelu, stride=2, pad=(0,1,0,1)) |>
    ConvTranspose((3,3), 8=>1, leakyrelu, stride=2, pad=(0,1,0,1)) #|> size
# (64,64,1,1)

#my data:
convT_ex = randn(9,9,1,1)
convTmodel = Chain(
    Conv((3,3), 1=>8, leakyrelu, stride=2, pad=(1,1)),
    ConvTranspose((3,3), 8=>1, leakyrelu, stride=2, pad=(1,1))
)

convTmodel(convT_ex) #no loss of pixels because filter(3x3) and image is 9x9


A = rand(1:3, 3, 3, 3)
findall(x->x==2, A)
