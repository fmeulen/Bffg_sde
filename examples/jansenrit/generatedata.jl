#Random.seed!(5)
#Random.seed!(15)

model= [:jr, :jr3][1]

if model == :jr
  θ0 =[3.25, 100.0, 22.0, 50.0, 95.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 2000.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
 # θ0 =[3.25, 100.0, 22.0, 50.0, 185.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 2000.0]  # this gives bimodality
 # θ0 =[3.25, 100.0, 22.0, 50.0, 530.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 5.0]  # also try this one
  ℙ0 = JansenRitDiffusion(θ0...)
  @show properties(ℙ0)
  AuxType = JansenRitDiffusionAux
end
if model == :jr3
  θ0 =[3.25, 100.0, 22.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 0.01, 2000.0, 1.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
  ℙ0 = JansenRitDiffusion3(θ0...)
  AuxType = JansenRitDiffusionAux3
end

#------  set observations
L = @SMatrix [0.0 1.0 -1.0 0.0 0.0 0.0]
m,  = size(L)
Σdiagel = 1e-7 # oorspr 1e-9
Σ = SMatrix{m,m}(Σdiagel*I)

#---- generate test data
T = 2.0 #1.0
x00 = @SVector zeros(6)
W = sample((-1.0):0.0001:T, wienertype(ℙ0))                        #  sample(tt, Wiener{ℝ{1}}())
Xf_prelim = solve(Euler(), x00, W, ℙ0)
# drop initial nonstationary behaviour
Xf = SamplePath(Xf_prelim.tt[10001:end], Xf_prelim.yy[10001:end])
x0 = Xf.yy[1]
dt = Xf.tt[2]-Xf.tt[1]

skipobs = 400# I took 400  all the time
obstimes =  Xf.tt[1:skipobs:end]
obsvals = map(x -> L*x, Xf.yy[1:skipobs:end])
pF = plot_all(ℙ0,  Xf, obstimes, obsvals)
savefig(joinpath(outdir, "forwardsimulated.png"))

#------- process observations, assuming x0 known
obs = [Observation(obstimes[1],  x0,  SMatrix{6,6}(1.0I), SMatrix{6,6}(Σdiagel*I))]
for i in 2:length(obstimes)
  push!(obs, Observation(obstimes[i], obsvals[i], L, Σ));
end

@show ℙ0

#----------- obs and obsvals are input to mcmc algorithm

# -- now obs with staionary prior on x0
obs = [Observation(-1.0,  x00,  SMatrix{6,6}(1.0I), SMatrix{6,6}(Σdiagel*I))]
for i in 1:length(obstimes)
  push!(obs, Observation(obstimes[i], obsvals[i], L, Σ));
end
pushfirst!(obsvals, SA[0.0])
Xf = Xf_prelim
x0 = x00


# remainder is checking 
S = DE(Vern7())

timegrids = set_timegrids(obs, 0.0005)
B = BackwardFilter(S, ℙ0, AuxType, obs, obsvals, timegrids) ;
Z = Innovations(timegrids, ℙ0);

# check
XX, ll = forwardguide(B, ℙ0)(x0, Z);

pG = plot_all(ℙ0, timegrids, XX)
l = @layout [a;b]
plot(pF, pG, layout=l)
savefig(joinpath(outdir,"forward_guidedinitial_separate.png"))

plot_all(ℙ0,Xf,  obstimes, obsvals, timegrids, XX)



savefig(joinpath(outdir,"forward_guidedinitial_overlaid.png"))


deviations = [obsvals[i] - L * XX[i-1][end]  for i in 2:length(obs)]
plot(obstimes[2:end], first.(deviations))
savefig(joinpath(outdir,"deviations_guidedinitial.png"))


TEST=false 
if TEST

# test
S = Vern7direct();
@time BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids);


S = DE(Vern7())
@time  BackwardFilter(S, ℙ, AuxType, obs,obsvals, timegrids);
# S = RK4()
# @btime  BackwardFilter(S, ℙ, AuxType, obs, timegrids);
@time forwardguide(x0, ℙ, Z, B);

# using Profile
# using ProfileView
# Profile.init()
# Profile.clear()
# S = DE(Vern7())#S = Vern7direct();
# @profile  BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids);
# @profile parupdate!(B, XX, movetarget, obs, obsvals, S, AuxType, timegrids; verbose=verbose)(x0, ℙ, Z, ll);# θ and XX may get overwritten
# ProfileView.view()
end