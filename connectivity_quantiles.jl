using Plots
pyplot()
using DelimitedFiles
using Statistics
using StatsBase

# Plot 1 - 9x9 example and scatterplot
observed = readdlm("data/csv/true_connect9x9_bear.csv")
predicted = readdlm("data/csv/model_connect9x9_bear.csv")
p1 = heatmap(observed; lab = false, leg = false, clim = (0, 1), axes=false, aspectratio=1, frame=:none, c=:Greens, dpi=180)
savefig("9_obs.png")
p2 = heatmap(predicted; lab = false, leg = false, clim = (0, 1), axes=false, aspectratio=1, frame=:none, c=:Purples, dpi=180)
savefig("9_prd.png")
scatter(observed, predicted, c = :black, xlim = (0, 1), ylim = (0, 1), lab="", leg=false, frame=:grid, dpi=180, aspectratio=1)
p3 = plot!([0, 1], [0, 1], c = :grey, ls = :dot)
savefig("9_sct.png")
plot(p1, p2, p3)

# Plot 2 - prediction for the black bear
observed = readdlm("data/csv/connectivity_oursnoir.csv")[1:1242,:]
predicted = readdlm("data/csv/mod_connectivity_bear.csv")
outside = findall(observed .== minimum(observed)) #findall points outside of data (ie. zeros)
observed[outside] .= NaN #set zeros to NaN
predicted[outside] .= NaN
mx = maximum(filter(!isnan, predicted)) #get the max value from predicted
predicted = predicted ./ mx #divide thru by max value to scale on between 0-1
q = (m) -> ecdf(filter(!isnan, m)) #create function, q, to return empirical cumulative distribution fnc (ecdf) for values that are not NaN
pr_q = q(predicted).(predicted) #apply fnc to transform output of NN into connectivity quantiles
ob_q = q(observed).(observed)
heatmap(pr_q, aspectratio=1, c=:BuPu, axis=nothing, frame=:none, leg=false, dpi=180)
savefig("bear_predicted.png")
heatmap(ob_q, aspectratio=1, c=:YlGn, axis=nothing, frame=:none, leg=false, dpi=180)
savefig("bear_observed.png")
histogram2d(vec(ob_q), vec(pr_q), aspectratio=1, c=:alpine, ylim=(0,1), xlim=(0,1), bins=50, leg=false, dpi=180)
savefig("hist2d.png")

# Plot 3 - difference
heatmap(pr_q .- ob_q, c=:PuOr, clim=(-1,1), dpi=180, aspectratio=1, leg=false, axis=nothing, frame=:none)
savefig("diff.png")
