wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
#using ForwardDiff
using DifferentialEquations
using Setfield
using Plots
using ConstructionBase
using Interpolations
using IterTools
using ProfileView
using RCall
using SparseArrays
using Parameters

import Bridge: R3, IndexedTime, llikelihood, kernelr3, constdiff, Euler, solve, solve!
import ForwardDiff: jacobian

sk = 0 # skipped in evaluating loglikelihood

include("jansenrit.jl")
include("jansenrit3.jl")

include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/tableaus_ode_solvers.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/types.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/forwardguiding.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/backwardfiltering.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/utilities.jl")

include("plotting.jl")

################################  TESTING  ################################################

S = DE(Vern7())

include("generatedata.jl")

timegrids = set_timegrids(obs, 0.0005)


iterations = 2_000  #5_00
skip_it = 200
subsamples = 0:skip_it:iterations # for saving paths

# define priors
priorC = Exponential(150.0)
priorσy = Normal(1500.0, 500.0)#Uniform(100.0, 3_000.0)

# define parameter moves
moveC = ParMove([:C], parameterkernel((short=[3.0], long=[10.0]); s=0.0), priorC, false)#, x-> SA[x.C])

moveσy = ParMove([:σy], parameterkernel((short=[3.0], long=[10.0]); s=0.0), priorσy, true)#, x-> SA[x.σy])

moveCᵒ = ParMove([:C], parameterkernel((short=[40.0], long=[100.0])), priorC, false)#, x-> SA[x.C])

moveCσy = ParMove([:C, :σy], parameterkernel((short=[2.0, 10.0], long=[10.0, 10.0]); s=0.0), product_distribution([priorC, priorσy]), true)#, x-> SA[x.C, x.σy])


# a small program

# settings
verbose = true # if true, surpress output written to console


θinit = 40.0
ESTσ = true



if ESTσ
  θ = (C=copy(θinit), σy = 1000.0)
  movetarget = moveCσy
  allparnames = [:C, :σy]
else
  θ = (; C = copy(θinit) ) # initial value for parameter
  movetarget = moveC
  allparnames = [:C]
end
ℙ = setproperties(ℙ0, θ)


𝒯 = 5_000.0 # temperature
ℙe = setproperties(ℙ0, C=copy(θinit),  σy = 𝒯)
allparnamese = [:C]
move_exploring = moveCᵒ



# pcn pars 
ρ = 0.95
ρe = 0.95

# initialisation of target chain 
B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids);
Z = Innovations(timegrids, ℙ);
Zbuffer = deepcopy(Z)
Zᵒ = deepcopy(Z)
ρs = fill(ρ, length(timegrids))
XX, ll = forwardguide(B, ℙ)(x0, Z);


# initialisation of exploring chain 
Be = BackwardFilter(S, ℙe, AuxType, obs, obsvals, timegrids);
Ze = Innovations(timegrids, ℙe);# deepcopy(Z);
Zeᵒ = deepcopy(Ze)
ρse = fill(ρe, length(timegrids))
XXe, lle = forwardguide(Be, ℙe)(x0, Ze);



θsave = [copy(getpar(allparnames, ℙ))]
XXsave = [copy(XX)]
llsave = [ll]

XXesave = [copy(XXe)]



accinnov = 0
accpar = 0
accinnove = 0
accpare = 0
accmove = 0


exploring = [State(x0, copy(Ze), getpar(allparnamese,ℙe), copy(lle))]

for i in 1:iterations
  (i % 500 == 0) && println(i)
  
  # update exploring chain
  
  lle, Be, ℙe, accpare_ = parupdate!(Be, XXe, move_exploring, obs, obsvals, S, AuxType, timegrids; verbose=verbose)(x0, ℙe, Ze, lle);# θe and XXe may get overwritten
  lle, accinnove_ = pcnupdate!(Be, ℙe, XXe, Zbuffer, Zeᵒ, ρse)(x0, Ze, lle); # Z and XX may get overwritten
  push!(exploring, State(x0, copy(Ze), getpar(allparnamese, ℙe), copy(lle)))   # collection of samples from exploring chain
  
  # update target chain
  smallworld = rand() > 0.33
  if smallworld
    ll, B, ℙ, accpar_ = parupdate!(B, XX, movetarget, obs, obsvals, S, AuxType, timegrids; verbose=verbose)(x0, ℙ, Z, ll);# θ and XX may get overwritten
    accmove_ =0
  else
    w = sample(exploring)     # randomly choose from samples of exploring chain
    ll, ℙ,  accmove_ = exploremoveσfixed!(B, Be, ℙe, move_exploring, XX, Zᵒ, w; verbose=verbose)(x0, ℙ, Z, ll) 
    accpar_ = 0
    #println(ℙ.C ==θ[1])
  end  
  ll, accinnov_ = pcnupdate!(B, ℙ, XX, Zbuffer, Zᵒ, ρs)(x0, Z, ll); # Z and XX may get overwritten

  
  # update acceptance counters
  accpar += accpar_; accpare += accpare_; accinnove += accinnove_; accinnov += accinnov_; accmove += accmove_
  # saving iterates
  push!(θsave, getpar(allparnames, ℙ))
  push!(llsave, ll)
  (i in subsamples) && push!(XXsave, copy(XX))
  (i % 500 == 0) && push!(XXesave, XXe)
  
  adjust_PNCparamters!(ρs, ρ)
end




# final imputed path
plot_all(ℙ, timegrids, XXsave[end])
plot_all(ℙ, Xf, obstimes, obsvals, timegrids, XXsave[end])
savefig(joinpath(outdir,"guidedpath_finaliteration.png"))

#
println("Target chain: accept% innov ", 100*accinnov/iterations,"%")
println("Target chain: accept% par ", 100*accpar/iterations,"%")
println("Exploring chain: accept% innov ", 100*accinnove/iterations,"%")
println("Exploring chain: accept% par ", 100*accpare/iterations,"%")

println("accept% swap ", 100*accmove/iterations,"%")

θesave = getindex.(getfield.(exploring,:θ),1)
llesave = getfield.(exploring, :ll)

h1 = histogram(getindex.(θsave,1),bins=35, label="target chain")
h2 = histogram(getindex.(θesave,1),bins=35, label="exploring chain")
plot(h1, h2, layout = @layout [a b])  

p1 = plot(llsave, label="target",legend=:bottom)    
plot!(p1,llesave, label="exploring")    
savefig(joinpath(outdir,"logliks.png"))

# traceplots
pa = plot(getindex.(θsave,1), label="target", legend=:top)
hline!(pa, [ℙ0.C], label="",color=:black)
plot!(pa, getindex.(θesave,1), label="exploring")


pb = plot(getindex.(θsave,2), label="target", legend=:top)
hline!(pb, [ℙ0.σy], label="",color=:black)
plot(pa, pb, layout = @layout [a; b])  
#savefig(joinpath(outdir,"traceplots.png"))






