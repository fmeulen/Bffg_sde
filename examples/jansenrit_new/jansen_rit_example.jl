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

import Bridge: R3, IndexedTime, llikelihood, kernelr3, constdiff, Euler, solve, solve!
import ForwardDiff: jacobian

sk = 0 # skipped in evaluating loglikelihood

include("jansenrit.jl")
include("jansenrit3.jl")

include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src_new/types.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src_new/forwardguiding.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src_new/backwardfiltering.jl")

include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src_new/utilities.jl")
include("plotting.jl")

################################  TESTING  ################################################

include("generatedata.jl")





# a small program

# settings
verbose = true

pars = ParInfo([:C], [false])
θ = [20.0] # initial value for parameter
θe = [20.0]

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
𝒯 = 10.0 # temperature
Ke = parameterkernel((short=[10.0], long=[60.0]))  

ℙe = setproperties(ℙ, σy = 𝒯*ℙ.σy)


# target chain 
B = BackwardFilter(ℙ, AuxType, obs, timegrids, x0, false);
Z = Innovations(timegrids, ℙ);
Zbuffer = deepcopy(Z)
Zᵒ = deepcopy(Z)
ρs = fill(ρ, length(timegrids))
XX, ll = forwardguide(B, ℙ, pars)(x0, θ, Z);

# exploring chain 
Be = BackwardFilter(ℙe, AuxType, obs, timegrids, x0, false);
Ze = Innovations(timegrids, ℙ);
Zbuffere = deepcopy(Ze)
Zeᵒ = deepcopy(Ze)
ρse = fill(ρe, length(timegrids))



XXe, lle = forwardguide(Be, ℙe, pars)(x0, θe, Ze);



θsave = [copy(θ)]
XXsave = [copy(XX)]
llsave = [ll]

θesave = [copy(θe)]
XXesave = [copy(XXe)]
llesave = [lle]



accinnov = 0
accpar = 0
accinnove = 0
accpare = 0
accswap = 0


exploring = []

for i in 1:iterations
  (i % 500 ==0) && println(i)
  targetshortproposal = rand()>0.33

  #exploring chain
  lle, accpare_ = parupdate!(Be, ℙe, pars, XXe, Ke, Prior; verbose=verbose)(x0, θe, Ze, lle);# θe and XXe may get overwritten
  lle, accinnove_ = pcnupdate!(Be, ℙe, pars, XXe, Zbuffer, Zeᵒ, ρse)(x0, θe, Ze, lle); # Z and XX may get overwritten

  # target chain
  ll, accpare_ = parupdate!(B, ℙ, pars, XX, K, Prior; verbose=verbose)(x0, θ, Z, ll);# θ and XX may get overwritten
  ll, accinnov_ = pcnupdate!(B, ℙ, pars, XX, Zbuffer, Zᵒ, ρs)(x0, θ, Z, ll); # Z and XX may get overwritten

  # saving iterates
  push!(θesave, copy(θe))
  push!(llesave, lle)
  push!(exploring, State(x0, deepcopy(Ze), deepcopy(θe), copy(lle)))
  push!(θsave, copy(θ))
  push!(llsave, ll)
  (i in subsamples) && push!(XXsave, copy(XX))

  adjust_PNCparamters!(ρs, ρ)

  # swap move
  
 if i>200
    
    w = sample(exploring)     # Randomly choose from samples of exploring chain
    # checkstate(Be, ℙe, pars)(w)
    copy!(Zᵒ, w.Z) # proppose from exploring chain in target chain
    θᵒ = copy(w.θ)
    XXᵒ, llᵒ = forwardguide(B, ℙ, pars)(x0, θᵒ, Zᵒ);
    # compute log proposalratio
    _, llproposal = forwardguide(Be, ℙe, pars)(x0, θ, Z);
    #_, llproposalᵒ = forwardguide(Be, ℙe, pars)(x0, θᵒ, Zᵒ);
    llproposalᵒ = w.ll
    A = llᵒ -ll + llproposal - llproposalᵒ 
    if log(rand()) < A
      @. XX = XXᵒ
      copy!(Z, Zᵒ)
      ll = llᵒ
      @. θ = θᵒ
      accswap +=1
      !verbose && print("✓")  
    end
    push!(θsave, copy(θ))
end

end
θsave
  
plot(map(x->x[1],θsave), label="target")
plot!(map(x->x[1],θesave), label="exploring")


plot_all(ℙ, timegrids, XXsave[end])
plot_all(ℙ, Xf, obstimes, obsvals, timegrids, XXsave[end])

println("accept% innov ", 100*accinnov/iterations,"%")
println("accept% par ", 100*accpar/iterations,"%")
println("accept% swap ", 100*accswap/iterations,"%")

histogram(map(x->x[1], θsave),bins=35)

p1 = plot(llsave, label="target")    
plot!(p1,llesave, label="target")    
llsave_last = llsave[500:end]
p2 = plot(500:length(llsave), llsave_last, label="target")    
plot(p1, p2, layout = @layout [a b])  



plot(map(x->x[1],θsave), label="target")
plot!(map(x->x[1],θesave), label="exploring")

