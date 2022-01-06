# Implementation of bridge simulation for the  NCLAR(3)-model
# Example 4.1 in https://arxiv.org/pdf/1810.01761v1.pdf
# Note that this implementation uses the ODEs detailed in "Continuous-discrete smoothing of diffusions"

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
using ForwardDiff
using DifferentialEquations


import Bridge: R3, IndexedTime, llikelihood, kernelr3, constdiff, solve, solve!, Euler
import ForwardDiff: jacobian

wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")
include("nclar.jl")

include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/funcdefs.jl")


################################  TESTING  ################################################
# settings sampler
iterations = 5_000 # 5*10^4
skip_it = 500  #1000
subsamples = 0:skip_it:iterations

T = .5
dt = 1/500
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

sk = 0 # skipped in evaluating loglikelihood


ℙ = NclarDiffusion(6.0, 2pi, 1.0)  # original setting
#ℙ = NclarDiffusion(16.0, 2pi, 4.0) 
x0 = ℝ{3}(0.0, 0.0, 0.0)
ℙ̃ = NclarDiffusionAux(ℙ.α, ℙ.ω, ℙ.σ)

# set observatins scheme 
easy_conditioning = true
obs_scheme =["full","firstcomponent"][2]

if obs_scheme=="full"
    LT = SMatrix{3,3}(1.0I)
    vT = easy_conditioning ?  ℝ{3}(1/32,1/4,1) :  ℝ{3}(5/128,3/8,2)
end
if obs_scheme=="firstcomponent"
    LT = @SMatrix [1. 0. 0.]
    vT = easy_conditioning ? ℝ{1}(1/32) : ℝ{1}(5/128)
end
#vT =  ℝ{1}(5.0)

m,  = size(LT)

Σdiagel = 10e-9
ΣT = SMatrix{m,m}(Σdiagel*I)


ρ = obs_scheme=="full" ? 0.85 : 0.99


# solve Backward Recursion
ϵ = 10e-2  
Hinit, Finit, Cinit =  init_HFC(vT, LT, dim(ℙ); ϵ=ϵ)
Hobs, Fobs, Cobs = observation_HFC(vT, LT, ΣT)

HT, FT, CT = fusion_HFC((Hinit, Finit, Cinit), (Hobs, Fobs, Cobs))
PT, νT, CT = convert_HFC_to_PνC(HT,FT,CT)

solv = DE(Tsit5())
solv = DE(Vern7())

𝒫 = PBridge(RK4(), ℙ, ℙ̃, tt, PT, νT, CT)
𝒫2 = PBridge(solv, ℙ, ℙ̃, tt, PT, νT, CT)
𝒫3 = PBridge(AssumedDensityFiltering(Tsit5()), ℙ, ℙ̃, tt, PT, νT, CT)

𝒫HFC = PBridge_HFC(RK4(), ℙ, ℙ̃, tt, HT, FT, CT)
𝒫HFC2 = PBridge_HFC(solv, ℙ, ℙ̃, tt, HT, FT, CT)

hcat(𝒫2.ν, 𝒫3.ν)
hcat(𝒫HFC.F, 𝒫HFC2.F)

# check
𝒫HFC.H[1] * 𝒫.P[1]
𝒫HFC2.H[1] * 𝒫2.P[1]

𝒫 = 𝒫3 # 𝒫HFC2
𝒫 = 𝒫HFC2


####################### MH algorithm ###################
W = sample(tt, Wiener())  #  sample(tt, Wiener{ℝ{3}}())
X = solve(Euler(), x0, W, ℙ)
Xᵒ = copy(X)
solve!(Euler(),Xᵒ, x0, W, 𝒫)
solve!(Euler(),X, x0, W, 𝒫)
ll = llikelihood(Bridge.LeftRule(), X, 𝒫, skip=sk)


# 𝒫 = PBridge(solv,ℙ, ℙ̃, tt, PT, νT, CT);
# 𝒫X = PBridge(ℙ, ℙ̃, tt, PT, νT, CT, X);
# hcat(𝒫.ν, 𝒫X.ν)


#𝒫 = 𝒫X
# using Plots
# plot(X.tt, getindex.(X.yy,1))


# Fold = 𝒫.F
# Hold = 𝒫.H
# # new constructor, adaptive
#𝒫 = PBridge(ℙ, ℙ̃, tt, FT, HT, CT, X)
 solve!(Euler(),Xᵒ, x0, W, 𝒫)
 solve!(Euler(),X, x0, W, 𝒫)
 ll = llikelihood(Bridge.LeftRule(), X, 𝒫, skip=sk)

# further initialisation
Wᵒ = copy(W)
W2 = copy(W)
XX = Any[]
if 0 in subsamples
    push!(XX, copy(X))
end

acc = 0

for iter in 1:iterations
    # ℙroposal
    global ll, acc, 𝒫
    sample!(W2, Wiener())
    #ρ = rand(Uniform(0.95,1.0))
    Wᵒ.yy .= ρ*W.yy + sqrt(1.0-ρ^2)*W2.yy
    solve!(Euler(),Xᵒ, x0, Wᵒ, 𝒫)


    llᵒ = llikelihood(Bridge.LeftRule(), Xᵒ, 𝒫,skip=sk)
    print("ll $ll $llᵒ, diff_ll: ",round(llᵒ-ll;digits=3))

    if log(rand()) <= llᵒ - ll
        X.yy .= Xᵒ.yy
        W.yy .= Wᵒ.yy
        ll = llᵒ
        print("✓")
        acc +=1
     

    end

println(vT - LT * X.yy[end])

    # if iter==1000
    #     𝒫 = PBridge(ℙ, ℙ̃, tt, PT, νT, CT, X)
    #     ll = llikelihood(Bridge.LeftRule(), X, 𝒫, skip=sk)
    # end

    println()
    if iter in subsamples
        push!(XX, copy(X))
    end
end

@info "Done."*"\x7"^6



include("process_output.jl")








