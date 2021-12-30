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


import Bridge: R3, IndexedTime, llikelihood, kernelr3, constdiff, Euler, solve, solve!
import ForwardDiff: jacobian

wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")
include("jansenrit.jl")

include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/funcdefs.jl")


################################  TESTING  ################################################
# settings sampler
iterations = 3_000 # 5*10^4
skip_it = 500  #1000
subsamples = 0:skip_it:iterations

T = 3.4
dt = 1/500
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

sk = 0 # skipped in evaluating loglikelihood

θtrue =[3.25, 100.0, 22.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 2000.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
θtrue =[0.0, 100.0, 0.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 00.0, 2000.0]  
θtrue =[3.25, 1.0, 22.0, 0.5, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 20.0]  # adjust a and b

ℙ = JansenRitDiffusion(θtrue...)
x0 = @SVector [0.08, 18.0, 15.0, -0.5, 0.0, 0.0] 
x0 = @SVector zeros(dim(ℙ))
ℙ̃ = JansenRitDiffusionAux(ℙ.a, ℙ.b , ℙ.A , ℙ.μy, ℙ.σy, T)


Random.seed!(4)
W = sample(tt, Wiener())                        #  sample(tt, Wiener{ℝ{1}}())
Xf = solve(Euler(), x0, W, ℙ)
#plot(X.tt, getindex.(X.yy,2))

LT = @SMatrix [0.0 1.0 -1.0 0.0 0.0 0.0]
xT = Xf.yy[end]
vT = LT * xT

m,  = size(LT)

Σdiagel = 10e-6
ΣT = SMatrix{m,m}(Σdiagel*I)


ρ = 0.9


# solve Backward Recursion
ϵ = 10e-2  
Hinit, Finit, Cinit =  init_HFC(vT, LT, dim(ℙ); ϵ=ϵ)
Hobs, Fobs, Cobs = observation_HFC(vT, LT, ΣT)

HT, FT, CT = fusion_HFC((Hinit, Finit, Cinit), (Hobs, Fobs, Cobs))
PT, νT, CT = convert_HFC_to_PνC(HT,FT,CT)

𝒫 = PBridge(RK4(), ℙ, ℙ̃, tt, PT, νT, CT)
𝒫2 = PBridge(DE(Tsit5()), ℙ, ℙ̃, tt, PT, νT, CT)
𝒫3 = PBridge(AssumedDensityFiltering(Tsit5()), ℙ, tt, PT, νT, CT)

𝒫HFC = PBridge_HFC(RK4(), ℙ, ℙ̃, tt, HT, FT, CT)
𝒫HFC2 = PBridge_HFC(DE(Tsit5()), ℙ, ℙ̃, tt, HT, FT, CT)
𝒫HFC3 = PBridge_HFC(DE(Vern7()), ℙ, ℙ̃, tt, HT, FT, CT)


hcat(𝒫.ν, 𝒫2.ν)
hcat(𝒫HFC2.F, 𝒫HFC3.F)


𝒫 = 𝒫HFC3

#𝒫 = PBridge(Adaptive(), ℙ, ℙ̃, tt, PT, νT, CT, X)

####################### MH algorithm ###################

X = solve(Euler(), x0, W, ℙ)
Xᵒ = copy(X)
solve!(Euler(),Xᵒ, x0, W, 𝒫)
solve!(Euler(),X, x0, W, 𝒫)
ll = llikelihood(Bridge.LeftRule(), X, 𝒫, skip=sk)

W = sample(tt, Wiener())  #  sample(tt, Wiener{ℝ{3}}())
solve!(Euler(),X, x0, W, ℙ̃)
#solve!(Euler(),X, x0, W, ℙ)
l = @layout [a b c ; d e f]
p1 = plot(X.tt, getindex.(X.yy,1))
p2 = plot(X.tt, getindex.(X.yy,2))
p3 = plot(X.tt, getindex.(X.yy,3))
p4 = plot(X.tt, getindex.(X.yy,4))
p5 = plot(X.tt, getindex.(X.yy,5))
p6 = plot(X.tt, getindex.(X.yy,6))
plot(p1,p2,p3,p4,p5,p6, layout=l)


LT*X.yy[end] - vT
using Plots
p = plot(X.tt, getindex.(X.yy,2) - getindex.(X.yy,3))
plot!(p, Xf.tt, getindex.(Xf.yy,2) - getindex.(Xf.yy,3))

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


k =5
p = plot(X.tt, getindex.(X.yy,k))
plot!(p, Xf.tt, getindex.(Xf.yy,k))




