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


################################  TESTING  ################################################
# settings sampler
iterations = 3_000 # 5*10^4
skip_it = 500  #1000
subsamples = 0:skip_it:iterations

T = 0.5
dt = 1/500
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

sk = 0 # skipped in evaluating loglikelihood

const d =2

# specify observation scheme
LT = @SMatrix [1. 0.]
Σdiagel = 10^(-10)
ΣT = @SMatrix [Σdiagel]

# specify target process
struct FitzhughDiffusion <: ContinuousTimeProcess{ℝ{2}}
    ϵ::Float64
    s::Float64
    γ::Float64
    β::Float64
    σ::Float64
end

Bridge.b(t, x, ℙ::FitzhughDiffusion) = ℝ{2}((x[1]-x[2]-x[1]^3+ℙ.s)/ℙ.ϵ, ℙ.γ*x[1]-x[2] +ℙ.β)
Bridge.σ(t, x, ℙ::FitzhughDiffusion) = ℝ{2}(0.0, ℙ.σ)
Bridge.constdiff(::FitzhughDiffusion) = true

ℙ = FitzhughDiffusion(0.1, 0.0, 1.5, 0.8, 0.3) # Ditlevsen-Samson
#P = FitzhughDiffusion(0.1, 0.0, 1.5, 0.8, 1.0)
x0 = ℝ{2}(-0.5, -0.6)

# Generate Data
# if generate_data
#      include("/Users/Frank/Sync/DOCUMENTS/onderzoek/code/diffbridges/truepaths_fh.jl")
# end

# specify auxiliary process
struct FitzhughDiffusionAux <: ContinuousTimeProcess{ℝ{2}}
    ϵ::Float64
    s::Float64
    γ::Float64
    β::Float64
    σ::Float64
    t::Float64
    u::Float64
    T::Float64
    v::Float64
end

function uv(t, ℙ::FitzhughDiffusionAux)
    λ = (t - ℙ.t)/(ℙ.T - ℙ.t)
    ℙ.v*λ + ℙ.u*(1-λ)
end

for k1 in (1:3)
    for k2 in (1:2)
        Random.seed!(4)# this is what i used all the time
        Random.seed!(44)
        aux_choice = ["linearised_end" "linearised_startend"  "matching"][k1]
        endpoint = ["first", "extreme"][k2]

        # settings sampler
        iterations =  !(k1==3) ? 5*10^4 : 10*10^4
        skip_it = 1000
        subsamples = 0:skip_it:iterations
        printiter = 100

        if endpoint == "first"
            #v = -0.959
            vT = SVector{1}(-1.0)
        elseif endpoint == "extreme"
            #v = 0.633
            vT = SVector{1}(1.1)
        else
            error("not implemented")
        end

        if aux_choice=="linearised_end"
            Bridge.B(t, ℙ::FitzhughDiffusionAux) = @SMatrix [1/ℙ.ϵ-3*ℙ.v^2/ℙ.ϵ  -1/ℙ.ϵ; ℙ.γ -1.0]
            Bridge.β(t, ℙ::FitzhughDiffusionAux) = ℝ{2}(ℙ.s/ℙ.ϵ+2*ℙ.v^3/ℙ.ϵ, ℙ.β)
            ρ = endpoint=="extreme" ? 0.9 : 0.0
        elseif aux_choice=="linearised_startend"
            Bridge.B(t, P::FitzhughDiffusionAux) = @SMatrix [1/P.ϵ-3*uv(t, ℙ)^2/P.ϵ  -1/ℙ.ϵ; ℙ.γ -1.0]
            Bridge.β(t, ℙ::FitzhughDiffusionAux) = ℝ{2}(ℙ.s/ℙ.ϵ+2*uv(t, ℙ)^3/ℙ.ϵ, ℙ.β)
            ρ = endpoint=="extreme" ? 0.98 : 0.0
        else
            Bridge.B(t, ℙ::FitzhughDiffusionAux) = @SMatrix [1/ℙ.ϵ  -1/ℙ.ϵ; ℙ.γ -1.0]
            Bridge.β(t, ℙ::FitzhughDiffusionAux) = ℝ{2}(ℙ.s/ℙ.ϵ-(ℙ.v^3)/ℙ.ϵ, ℙ.β)
            ρ = 0.99
        end

        Bridge.σ(t, ℙ::FitzhughDiffusionAux) = ℝ{2}(0.0, ℙ.σ)
        Bridge.constdiff(::FitzhughDiffusionAux) = true

        Bridge.b(t, x, ℙ::FitzhughDiffusionAux) = Bridge.B(t,ℙ) * x + Bridge.β(t,ℙ)
        Bridge.a(t, ℙ::FitzhughDiffusionAux) = Bridge.σ(t,ℙ) * Bridge.σ(t, ℙ)'

        ℙ̃ = FitzhughDiffusionAux(ℙ.ϵ, ℙ.s, ℙ.γ, ℙ.β, ℙ.σ, tt[1], x0[1], tt[end], vT[1])
    end
end

# solve Backward Recursion
ϵ = 10e-2  

Hinit, Finit, Cinit =  init_HFC(vT, LT; ϵ=ϵ)
Hobs, Fobs, Cobs = observation_HFC(vT, LT, ΣT)

HT, FT, CT = fusion_HFC((Hinit, Finit, Cinit), (Hobs, Fobs, Cobs))
PT, νT, CT = convert_HFC_to_PνC(HT,FT,CT)

𝒫 = PBridge(ℙ, ℙ̃, tt, PT, νT, CT);

####################### MH algorithm ###################
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


ρ = 0.5
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






