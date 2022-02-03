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

include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/vern7.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/types.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/forwardguiding.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/backwardfiltering.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/utilities.jl")

include("plotting.jl")

################################  TESTING  ################################################
S = Vern7direct()  # solver for backward ODEs

include("generatedata.jl")


# a small program

# settings
verbose = true # if true, surpress output written to console

pars = ParInfo([:C], [false])
θinit = 30.0
θ = [θinit+400.0] # initial value for parameter
θe = [θinit]

timegrids = set_timegrids(obs, 0.0005)

iterations = 5_000
skip_it = 200
subsamples = 0:skip_it:iterations # for saving paths

Prior = Exponential(150.0)

# target chain 
ρ = 0.95
K = parameterkernel((short=[2.0], long=[10.0]); s=0.0) # always use short-range proposal kernel  

# exploring chain
ρe = 0.95
𝒯 = 2.0 # temperature
Ke = parameterkernel((short=[10.0], long=[100.0]))  

ℙe = setproperties(ℙ, σy = 𝒯*ℙ.σy)


# initialisation of target chain 
B = BackwardFilter(S, ℙ, AuxType, obs, timegrids, x0, false);
Z = Innovations(timegrids, ℙ);
Zbuffer = deepcopy(Z)
Zᵒ = deepcopy(Z)
ρs = fill(ρ, length(timegrids))
XX, ll = forwardguide(B, ℙ, pars)(x0, θ, Z);

# initialisation of exploring chain 
Be = BackwardFilter(S, ℙe, AuxType, obs, timegrids, x0, false);
Ze = Innovations(timegrids, ℙ);
Zeᵒ = deepcopy(Ze)
ρse = fill(ρe, length(timegrids))
XXe, lle = forwardguide(Be, ℙe, pars)(x0, θe, Ze);



θsave = [copy(θ)]
XXsave = [copy(XX)]
llsave = [ll]

XXesave = [copy(XXe)]



accinnov = 0
accpar = 0
accinnove = 0
accpare = 0
accmove = 0


exploring = State[]

for i in 1:iterations
  (i % 500 == 0) && println(i)
  
  # update exploring chain
  lle, accpare_ = parupdate!(Be, ℙe, pars, XXe, Ke, Prior; verbose=verbose)(x0, θe, Ze, lle);# θe and XXe may get overwritten
  lle, accinnove_ = pcnupdate!(Be, ℙe, pars, XXe, Zbuffer, Zeᵒ, ρse)(x0, θe, Ze, lle); # Z and XX may get overwritten
  push!(exploring, State(x0, copy(Ze), copy(θe), copy(lle)))   # collection of samples from exploring chain

  # update target chain
  smallworld = rand() >0.33
  if smallworld
    ll, accpar_ = parupdate!(B, ℙ, pars, XX, K, Prior; verbose=verbose)(x0, θ, Z, ll);# θ and XX may get overwritten
    accmove_ =0
  else
    w = sample(exploring)     # randomly choose from samples of exploring chain
    ll, accmove_ = exploremove!(B, ℙ, Be, ℙe, XX, Zᵒ, w; verbose=true)(x0, θ, Z, ll) 
    accpar_ = 0
  end  
  ll, accinnov_ = pcnupdate!(B, ℙ, pars, XX, Zbuffer, Zᵒ, ρs)(x0, θ, Z, ll); # Z and XX may get overwritten

  # update acceptance counters
  accpar += accpar_; accpare += accpare_; accinnove += accinnove_; accinnov += accinnov_; accmove += accmove_
  # saving iterates
  push!(θsave, copy(θ))
  push!(llsave, ll)
  (i in subsamples) && push!(XXsave, copy(XX))
  (i % 500 == 0) && push!(XXesave, XXe)
  
  adjust_PNCparamters!(ρs, ρ)
end




# final imputed path
plot_all(ℙ, timegrids, XXsave[end])
plot_all(ℙ, Xf, obstimes, obsvals, timegrids, XXsave[end])

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

p1 = plot(llsave, label="target")    
plot!(p1,llesave, label="exploring")    

# traceplots
plot(getindex.(θsave,1), label="target")
plot!(getindex.(θesave,1), label="exploring")








