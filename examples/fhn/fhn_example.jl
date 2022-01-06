using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
using ForwardDiff


import Bridge: R3, IndexedTime, llikelihood, kernelr3, constdiff
import ForwardDiff: jacobian

wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")


include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/funcdefs.jl")

aux_choice = ["linearised_end" "linearised_startend"  "matching"][1]
include("fhn.jl")

################################  TESTING  ################################################
# settings sampler
iterations = 5_000 # 5*10^4
skip_it = 100  #1000
subsamples = 0:skip_it:iterations

T = 2.0 # original setting
#T = 5.0
dt = 1/500
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

sk = 0 # skipped in evaluating loglikelihood



# specify observation scheme
LT = @SMatrix [1. 0.]
Σdiagel = 10^(-9)
ΣT = @SMatrix [Σdiagel]

# specify target process

x0 = ℝ{2}(-0.5, -0.6)
endpoint = ["first", "extreme"][2]

if endpoint == "first"
    vT = SVector{1}(-1.0)
else endpoint == "extreme"
    vT = SVector{1}(1.1)
end


ℙ = FitzhughDiffusion(0.1, 0.0, 1.5, 0.8, 0.3) # Ditlevsen-Samson

ℙ̃ = FitzhughDiffusionAux(ℙ.ϵ, ℙ.s, ℙ.γ, ℙ.β, ℙ.σ, tt[1], x0[1], tt[end], vT[1])

# solve Backward Recursion
ϵ = 10e-2  

Hinit, Finit, Cinit =  init_HFC(vT, LT, dim(ℙ); ϵ=ϵ)
Hobs, Fobs, Cobs = observation_HFC(vT, LT, ΣT)

HT, FT, CT = fusion_HFC((Hinit, Finit, Cinit), (Hobs, Fobs, Cobs))
PT, νT, CT = convert_HFC_to_PνC(HT,FT,CT)

# 
#𝒫 = PBridge(RK4(),  ℙ, ℙ̃, tt, PT, νT, CT);
#𝒫 = PBridge(DE(Tsit5()),  ℙ, ℙ̃, tt, PT, νT, CT);
𝒫 = PBridge_HFC(DE(Vern7()), ℙ, ℙ̃, tt, HT, FT, CT)


####################### MH algorithm ###################
# alternatively, if σ is defined as a matrix, then set
if false
    Bridge.σ(t, x, ℙ::FitzhughDiffusion) = @SMatrix [0.0;  ℙ.σ]
    W = sample(tt, Wiener{ℝ{1}}())
    X = solve(Euler(), x0, W, ℙ)
end

W = sample(tt, Wiener())
X = solve(Euler(), x0, W, ℙ)
Xᵒ = copy(X)
solve!(Euler(),Xᵒ, x0, W, 𝒫)
solve!(Euler(),X, x0, W, 𝒫)
ll = llikelihood(Bridge.LeftRule(), X, 𝒫, skip=sk)

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


ρ = 0.95
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
        # 𝒫 = PBridge(ℙ, ℙ̃, tt, FT, HT, CT, X)
        # ll = llikelihood(Bridge.LeftRule(), X, 𝒫, skip=sk)

    end

    println()
    if iter in subsamples
        push!(XX, copy(X))
    end
end

@info "Done."*"\x7"^6



include("process_output.jl")






