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

import Bridge: R3, IndexedTime, llikelihood, kernelr3, constdiff, Euler, solve, solve!
import ForwardDiff: jacobian



include("jansenrit.jl")
include("jansenrit3.jl")

include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src_new/types.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src_new/forwardguiding.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src_new/backwardfiltering.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src_new/funcdefs.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src_new/utilities.jl")
include("plotting.jl")
include("generatedata.jl")
################################  TESTING  ################################################

sk = 0 # skipped in evaluating loglikelihood

# a small program

# settings
verbose = false

pars = ParInfo([:C], [false])
θ = [20.0] # initial value for parameter

timegrids = set_timegrids(obs, 0.0005)

iterations = 1_000
skip_it = 200
subsamples = 0:skip_it:iterations # for saving paths

# target chain 
ρ = 0.99
K = parameterkernel((short=[2.0], long=[10.0]); s=0.5)  

# exploring chain
ρℰ = 0.99
𝒯 = 10.0 # temperature
Kℰ = parameterkernel((short=[10.0], long=[20.0]); s=0.5)  
ℙℰ = setproperties(ℙ, σy = 𝒯*ℙ.σy)


# target chain 
B = BackwardFilter(AuxType, obs, timegrids, x0, false);
Z = Innovations(timegrids, ℙ);
Zbuffer = deepcopy(Z)
Zᵒ = deepcopy(Z)
ρs = fill(ρ, length(timegrids))
XX, ll = forwardguide(B, ℙ, pars)(x0, θ, Z);

# exploring chain 
Bℰ = B #BackwardFilter(AuxType, obs, timegrids, x0, false);
Zℰ = Innovations(timegrids, ℙ);
Zbufferℰ = deepcopy(Zℰ)
Zℰᵒ = deepcopy(Zℰ)
ρsℰ = fill(ρℰ, length(timegrids))
XXℰ, llℰ = forwardguide(B, ℙℰ, pars)(x0, θ, Zℰ);

θsave = [copy(θ)]
XXsave = [copy(XX)]
llsave = [ll]

accinnov = 0
accpar = 0

# updatepar!(B, ℙ, pars) = (x0, θ, Z, XX, ll) ->  
# accpar_ = update!(B, ℙ, pars)(x0, θ, Z, XX, ll)


# updateinnov!(B, ℙ, pars, Zbuffer, Zᵒ, ρs, ρ) = (x0, θ, Z, XX, ll) ->  
# accinnov_ = updateinnov!(B, ℙ, pars, Zbuffer, Zᵒ, ρs, ρ)(x0, θ, Z, XX, ll) 


for i in 1:iterations
  # target chain
  θᵒ = K(θ)  
  XXᵒ, llᵒ = forwardguide(B, ℙ, pars)(x0, θᵒ, Z);
  !verbose && printinfo(ll, llᵒ, "par") 
  if log(rand()) < llᵒ-ll
    XX, XXᵒ = XXᵒ, XX
    ll = llᵒ
    θ .= θᵒ
    accpar += 1
    !verbose && print("✓")  
  end
  push!(llsave, ll)

  pcn!(Zᵒ, Z, Zbuffer, ρs, ℙ)
  XXᵒ, llᵒ = forwardguide(B, ℙ, pars)(x0, θ, Zᵒ);
  !verbose && printinfo(ll, llᵒ, "pCN") 
    if log(rand()) < llᵒ-ll
      XX, XXᵒ = XXᵒ, XX
      copy!(Z, Zᵒ)
      # for i in eachindex(Z.z)
      #   Z.z[i].yy .= Zᵒ.z[i].yy 
      # end
    ll = llᵒ
    accinnov +=1
    !verbose && print("✓")  
  end

  push!(θsave, copy(θ))
  push!(llsave, ll)
  (i in subsamples) && push!(XXsave, copy(XX))

  adjust_PNCparamters!(ρs, ρ)

  # exploring chain

end
θsave

plot(map(x->x[1],θsave))

plot_all(ℙ, timegrids, XXsave[end])
plot_all(ℙ, Xf, obstimes, obsvals, timegrids, XXsave[end])

println("accept% innov ", 100*accinnov/iterations,"%")
println("accept% par ", 100*accpar/iterations,"%")


histogram(map(x->x[1], θsave),bins=35)

p1 = plot(llsave, label="target")    
llsave_last = llsave[500:end]
p2 = plot(500:length(llsave), llsave_last, label="target")    
plot(p1, p2, layout = @layout [a b])  



