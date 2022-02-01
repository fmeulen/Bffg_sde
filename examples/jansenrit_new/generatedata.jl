Random.seed!(5)

model= [:jr, :jr3][1]

if model == :jr
  θtrue =[3.25, 100.0, 22.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 2000.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
  θtrue =[3.25, 100.0, 22.0, 50.0, 185.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 2000.0]
  ℙ = JansenRitDiffusion(θtrue...)
  show(properties(ℙ))
  AuxType = JansenRitDiffusionAux
end
if model == :jr3
  θtrue =[3.25, 100.0, 22.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 0.01, 2000.0, 1.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
  ℙ = JansenRitDiffusion3(θtrue...)
  AuxType = JansenRitDiffusionAux3
end

#------  set observations
L = @SMatrix [0.0 1.0 -1.0 0.0 0.0 0.0]
m,  = size(L)
Σdiagel = 1e-9
Σ = SMatrix{m,m}(Σdiagel*I)

#---- generate test data
T = 1.0
x0 = @SVector zeros(6)
W = sample((-1.0):0.0001:T, wienertype(ℙ))                        #  sample(tt, Wiener{ℝ{1}}())
Xf_prelim = solve(Euler(), x0, W, ℙ)
# drop initial nonstationary behaviour
Xf = SamplePath(Xf_prelim.tt[10001:end], Xf_prelim.yy[10001:end])
x0 = Xf.yy[1]


skipobs = 400  #length(Xf.tt)-1 #500
obstimes =  Xf.tt[1:skipobs:end]
obsvals = map(x -> L*x, Xf.yy[1:skipobs:end])
pF = plot_all(ℙ,  Xf, obstimes, obsvals)
savefig(joinpath(outdir, "forwardsimulated.png"))

#------- process observations
obs = Observation[]
for i ∈ eachindex(obsvals)
    push!(obs, Observation(obstimes[i], obsvals[i], L, Σ))
end
obs[1] = Observation(obstimes[1], x0, SMatrix{6,6}(1.0I), SMatrix{6,6}(Σdiagel*I))


# remainder is checking 

timegrids = set_timegrids(obs, 0.0005)
B = BackwardFilter(ℙ, AuxType, obs, timegrids, x0, false);
Z = Innovations(timegrids, ℙ);

# check
#forwardguide(x0, ℙ, Z.z[1], B.Ms[1]);
XX, ll = forwardguide(x0, ℙ, Z, B);

pG = plot_all(ℙ, timegrids,XX)
l = @layout [a;b]
plot(pF, pG, layout=l)
savefig(joinpath(outdir,"forward_guidedinitial_separate.png"))

plot_all(ℙ,Xf, obstimes, obsvals, timegrids, XX)
savefig(joinpath(outdir,"forward_guidedinitial_overlaid.png"))

deviations = [obs[i].v - obs[i].L * XX[i-1][end]  for i in 2:length(obs)]
plot(obstimes[2:end], map(x-> x[1,1], deviations))
savefig(joinpath(outdir,"deviations_guidedinitial.png"))


