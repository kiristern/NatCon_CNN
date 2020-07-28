using Plots
pyplot()
using DelimitedFiles
using Statistics
using StatsBase

# Plot 1 - 9x9 example and scatterplot
observed = readdlm("true_connect9x9_bear.csv")
predicted = readdlm("model_connect9x9_bear.csv")
heatmap(observed; lab = false, leg = false, clim = (0, 1), axes=false, aspectratio=1, frame=:none, c=:Greens, dpi=180)
savefig("9_obs.png")
heatmap(predicted; lab = false, leg = false, clim = (0, 1), axes=false, aspectratio=1, frame=:none, c=:Purples, dpi=180)
savefig("9_prd.png")
scatter(observed, predicted, c = :black, xlim = (0, 1), ylim = (0, 1), lab="", leg=false, frame=:grid, dpi=180, aspectratio=1)
plot!([0, 1], [0, 1], c = :grey, ls = :dot)
savefig("9_sct.png")

# Plot 2 - prediction for the black bear
observed = readdlm("connectivity_oursnoir.csv")[1:1242,:]
predicted = readdlm("mod_connectivity_bear.csv")
outside = findall(observed .== minimum(observed))
observed[outside] .= NaN
predicted[outside] .= NaN
mx = maximum(filter(!isnan, predicted))
predicted = predicted ./ mx
q = (m) -> ecdf(filter(!isnan, m))
pr_q = q(predicted).(predicted)
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
