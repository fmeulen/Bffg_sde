using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
using ForwardDiff


import Bridge: R3, IndexedTime, llikelihood, kernelr3, constdiff
import ForwardDiff: jacobian

wdir = "/Users/frankvandermeulen/.julia/dev/Bffg_sde"
cd(wdir)
outdir= wdir * "/out/"



function kernelrk4(f, t, y, dt, ℙ)
    k1 = f(t, y, ℙ)
    k2 = f(t + 0.5*dt, y + 0.5*k1*dt, ℙ)
    k3 = f(t + 0.5*dt, y + 0.5*k2*dt, ℙ)
    k4 = f(t + dt, y + k3*dt, ℙ)
    y + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
end

function convert_HFC_to_PνC(H,F,C)
    P = inv(H)
    P, P*F, C
end   

function convert_PνC_to_HFC(P,ν,C)
    H = inv(P)
    P, P\ν, C
end   

vectorise(P,ν, C) = vcat(SVector(P), ν, SVector(C))


"""
    static_accessor_HFc(u::SVector, ::Val{T}) where T
Access data stored in the container `u` so that it matches the shapes of H,F,c
and points to the correct points in `u`. `T` is the dimension of the stochastic
process.
"""
function static_accessor_HFc(u::K, ::Val{T}) where {K<:Union{SVector,MVector},T}
    Hidx = SVector{T*T,Int64}(1:T*T)
    Fidx = SVector{T,Int64}((T*T+1):(T*T+T))
    reshape(u[Hidx], Size(T,T)), u[Fidx], u[T*T+T+1]
end








struct PBridge{T,Tℙ,Tℙ̃,TP,Tν,TC} <: ContinuousTimeProcess{T}
    ℙ::Tℙ   # diffusion 
    ℙ̃::Tℙ̃   # auxiliary process
    tt::Vector{Float64}  # time grid
    P::Vector{TP}        # P=ℙ values on time grid
    ν::Vector{Tν}        # ν values on time grid
    C::TC                # constant to compute h-function
    PBridge(ℙ::Tℙ, ℙ̃::Tℙ̃, tt, Pt::Vector{TP}, νt::Vector{Tν}, C::TC) where {Tℙ,Tℙ̃,TP,Tν,TC} =
        new{Bridge.valtype(ℙ),Tℙ,Tℙ̃,TP,Tν,TC}(ℙ, ℙ̃, tt, Pt, νt, C)

    # constructor: provide (timegrid, ℙ, ℙ̃, νT, PT, CT)    
    function PBridge(ℙ, ℙ̃, tt_, PT::TP, νT::Tν, CT) where {TP, Tν}
        tt = collect(tt_)
        N = length(tt)
        Pt = zeros(TP, N)
        νt = zeros(Tν, N)
        _, _, C = pbridgeode!(R3(), ℙ̃, tt, (Pt, νt), (PT, νT, CT))
        PBridge(ℙ, ℙ̃, tt, Pt, νt, C)
    end

    # 2nd constructor (use also samplepath X in backward ODEs): provide (timegrid, ℙ, νT, PT, CT, X::Sampleℙath)    
    function PBridge(ℙ, ℙ̃, tt_, PT::TP, νT::Tν, CT, X::SamplePath) where  {TP, Tν}
        tt = collect(tt_)
        N = length(tt)
        Pt = zeros(TP, N)
        νt = zeros(Tν, N)
        _, _, C = pbridgeode!(R3(), ℙ, tt, (Pt, νt), (PT, νT, CT), X)
        PBridge(ℙ, ℙ̃, tt, Pt, νt, C)
    end
end







function pbridgeode!(::R3, ℙ̃, t, (Pt, νt), (PT, νT, CT))
    access = Val{}(d)

    function dPνC(s, y, ℙ̃)
        access = Val{}(d)
        P, ν, _ = static_accessor_HFc(y, access)
        _B, _β, _σ, _a = Bridge.B(s, ℙ̃), Bridge.β(s, ℙ̃), Bridge.σ(s, ℙ̃), Bridge.a(s, ℙ̃)

        dP =  (_B * P) + (P * _B') - _a
        dν =  (_B * ν) + _β
        F = (P \ ν)
        dC = dot(_β, F) + 0.5*Bridge.outer(F' * _σ) - 0.5*tr( (P \ (_a)))
        vectorise(dP, dν, dC)
    end

    Pt[end] = PT
    νt[end] = νT
    C = CT

    y = vectorise(PT, νT, CT)
    println(νT)
    println()
    
    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        y = kernelrk4(dPνC, t[i+1], y, dt, ℙ̃)
        Pt[i], νt[i], C = static_accessor_HFc(y, access)
    end

    Pt, νt, C
end

P((i,t)::IndexedTime, x, 𝒫::PBridge) = 𝒫.P[i]
ν((i,t)::IndexedTime, x, 𝒫::PBridge) = 𝒫.ν[i]
r((i,t)::IndexedTime, x, 𝒫::PBridge) = (𝒫.P[i] \ (𝒫.ν[i] - x) )

function Bridge._b((i,t)::IndexedTime, x, 𝒫::PBridge)  
    Bridge.b(t, x, 𝒫.ℙ) + Bridge.a(t, x, 𝒫.ℙ) * r((i,t),x,𝒫)   # (𝒫.ν[i] - 𝒫.P[i]*x)
end


Bridge.σ(t, x, 𝒫::PBridge) = Bridge.σ(t, x, 𝒫.ℙ)
Bridge.a(t, x, 𝒫::PBridge) = Bridge.a(t, x, 𝒫.ℙ)
Bridge.Γ(t, x, 𝒫::PBridge) = Bridge.Γ(t, x, 𝒫.ℙ)
Bridge.constdiff(𝒫::PBridge) = Bridge.constdiff(𝒫.ℙ) && Bridge.constdiff(𝒫.ℙ̃)

function logh̃(x, 𝒫::PBridge) 
    H1, F1, C = convert_PνC_to_HFC(𝒫.P[1], 𝒫.ν[1],𝒫.C)
    -0.5 * x' * H1 * x + F1' * x - C    
end

function llikelihood(::LeftRule, X::SamplePath, 𝒫::PBridge; skip = 0)
    tt = X.tt
    xx = X.yy

    som::Float64 = 0.
    for i in 1:length(tt)-1-skip #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        r̃ = r((i,s), x, 𝒫)
        dt = tt[i+1]-tt[i]

        som += dot( Bridge._b((i,s), x, 𝒫.ℙ) - Bridge._b((i,s), x, 𝒫.ℙ̃), r̃) 
        if !constdiff(𝒫)
            P = P((i,s), x, 𝒫)
            som -= 0.5*tr( (a((i,s), x, 𝒫.ℙ) - a((i,s), x, 𝒫.ℙ̃)) * P )   
            som += 0.5 *( r̃'* ( a((i,s), x, 𝒫.ℙ) - a((i,s), x, 𝒫.ℙ̃) ) * r̃ ) 
        end
        som *= dt 
    end
    som + logh̃(X.yy[1], 𝒫)
end


"""
    init_HFC(v, L; ϵ=0.01)

    First computes xT = L^(-1) * vT (Moore-Penrose inverse)    
    Then convert artifical observation v ~ N(xT, ϵ^(-1) * I)
    to triplet  (H, F, C)
"""
function init_HFC(v, L; ϵ=0.01)
    P = ϵ^(-1)*SMatrix{3,3}(1.0I)
    xT = L\v
    z = zero(xT)
    C = -logpdf(Bridge.Gaussian(z, P), z) 
    convert_PνC_to_HFC(P, xT ,C)
end


"""
    observation_HFC(v, L, Σ)

    Convert observation v ~ N(Lx, Σ)
    to triplet  (H, F, C)
"""
function observation_HFC(v, L, Σ)
    A = L' * inv(Σ)
    H = A*L
    H, A*v, - logpdf(Bridge.Gaussian(zero(v), Σ), v)
end
    
"""
    fusion_HFC((H1, F1, C1), (H2, F2, C2))

    returns added characteristics that correspond to fusion in (H,F,C)-parametrisation
"""
function fusion_HFC((H1, F1, C1), (H2, F2, C2))
    H1 + H2, F1 + F2, C1+C2
end




################################################################################
################################  TESTING  ################################################
# settings sampler
iterations = 3_000 # 5*10^4
skip_it = 500  #1000
subsamples = 0:skip_it:iterations

T = 0.5
dt = 1/5000
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

sk = 0 # skipped in evaluating loglikelihood
const d=3

easy_conditioning = true
obs_scheme =["full","firstcomponent"][1]
ρ = obs_scheme=="full" ? 0.85 : 0.95
if obs_scheme=="full"
    LT = SMatrix{3,3}(1.0I)
    vT = easy_conditioning ?  ℝ{3}(1/32,1/4,1) :  ℝ{3}(5/128,3/8,2)
end
if obs_scheme=="firstcomponent"
    LT = @SMatrix [1. 0. 0.]
    vT = easy_conditioning ? ℝ{1}(1/32) : ℝ{1}(5/128)
end


ρ = 0.0

Σdiagel = 10e-9
m,  = size(LT)
ΣT = SMatrix{m,m}(Σdiagel*I)

# specify target process
struct NclarDiffusion <: ContinuousTimeProcess{ℝ{3}}
    α::Float64
    ω::Float64
    σ::Float64
end

Bridge.b(t, x, ℙ::NclarDiffusion) = ℝ{3}(x[2],x[3],-ℙ.α * sin(ℙ.ω * x[3]))
Bridge.σ(t, x, ℙ::NclarDiffusion) = ℝ{3}(0.0, 0.0, ℙ.σ)
Bridge.constdiff(::NclarDiffusion) = true

jacobianb(t, x, ℙ::NclarDiffusion) = @SMatrix [0.0 1.0 0.0 ; 0.0 0.0 1.0 ; 0.0 0.0 -ℙ.α * ℙ.ω * cos(ℙ.ω * x[3])]

ℙ = NclarDiffusion(6.0, 2pi, 1.0)
x0 = ℝ{3}(0.0, 0.0, 0.0)


# specify auxiliary process
struct NclarDiffusionAux <: ContinuousTimeProcess{ℝ{3}}
    α::Float64
    ω::Float64
    σ::Float64
end

Bridge.B(t, ℙ::NclarDiffusionAux) = @SMatrix [0.0 1.0 0.0 ; 0.0 0.0 1.0 ; 0.0 0.0 0.0]
Bridge.β(t, ℙ::NclarDiffusionAux) = ℝ{3}(0.0,0.0,0)
Bridge.σ(t,  ℙ::NclarDiffusionAux) = ℝ{3}(0.0,0.0, ℙ.σ)
Bridge.constdiff(::NclarDiffusionAux) = true
Bridge.b(t, x, ℙ::NclarDiffusionAux) = Bridge.B(t,ℙ) * x + Bridge.β(t,ℙ)
Bridge.a(t, ℙ::NclarDiffusionAux) = Bridge.σ(t,ℙ) * Bridge.σ(t,  ℙ)'
Bridge.a(t, x, ℙ::NclarDiffusionAux) = Bridge.a(t,ℙ) 

ℙ̃ = NclarDiffusionAux(ℙ.α, ℙ.ω, ℙ.σ)


# Solve Backward Recursion
ϵ = 10e-2  # choice not too important for bridges
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

acc = 0

for iter in 1:iterations
    # ℙroposal
    global ll, acc, 𝒫
    sample!(W2, Wiener())
    #ρ = rand(Uniform(0.95,1.0))
    Wᵒ.yy .= ρ*W.yy + sqrt(1-ρ^2)*W2.yy
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

# write mcmc iterates to csv file
extractcomp(v,i) = map(x->x[i], v)

iterates = [Any[s, tt[j], d, XX[i].yy[j][d]] for d in 1:3, j in 1:length(X), (i,s) in enumerate(subsamples) ][:]
df_iterates = DataFrame(iteration=extractcomp(iterates,1),time=extractcomp(iterates,2), component=extractcomp(iterates,3), value=extractcomp(iterates,4))
CSV.write(outdir*"iterates-"*obs_scheme*".csv",df_iterates)

ave_acc_perc = 100*round(acc/iterations;digits=2)





################################################################################

# do this later
if false 

    # solve ODEs using current Sampleℙath X (need to pass ℙ for that, instead of ℙ̃)
    function pbridgeode!(::R3, ℙ, t, (νt, Pt), (νT, PT, CT), X)
        Pt[end] = PT
        νt[end] = νT
        ν, P, C = νT, PT, CT


        # function dP(s, y, (ℙ,x)) 
        #     #ff(s,ℙ) = u -> Bridge.b(s,u,ℙ)
        #     #B = jacobian(u->ff(s,ℙ)(u), x)
        #     B = jacobianb(s,x,ℙ)
        #     out = - B'*y - y*B + y*Bridge.a(s, x, ℙ)*y'
        #     return out
        #  end
        
        dP(s, y, (B̃, ã)) = - B̃'*y - y * B̃ + y*ã*y'


        #dν(s, y, (P,ℙ̃)) = -Bridge.B(s, ℙ̃)'*y + P*Bridge.a(s, ℙ̃)*y  + P*Bridge.β(s, ℙ̃)
        # function dν(s, y, (P,ℙ,x)) 
        #     #B = jacobian(u->Bridge.b(s,u,ℙ),x)
        #     B = jacobianb(s,x,ℙ)
        #     out = -B'*y + P*Bridge.a(s, x, ℙ)*y + P*(Bridge.b(s,x,ℙ) - B*x)
        #     return out
        #  end

        dν(s,y, (B̃, ã, β̃)) = -B̃'*y + P*ã*y + P*β̃

        for i in length(t)-1:-1:1
            dt = t[i] - t[i+1]
            x = X.yy[i+1]
            s = t[i+1]

            B̃ = jacobianb(s,x,ℙ)
            β̃ = Bridge.b(s,x,ℙ) - B̃*x
            ã =  Bridge.a(s, x, ℙ)

            C += ( β̃'*ν + 0.5*ν'*ã*ν - 0.5*tr(P*ã) ) * dt
            #P = kernelr3(dP, t[i+1], P, dt, (ℙ,x))
            P = kernelrk4(dP, t[i+1], P, dt, (B̃, ã))
            #ν = kernelr3(dν, t[i+1], ν, dt, (P, ℙ, x))
            ν = kernelrk4(dν, t[i+1], ν, dt, (B̃, ã, β̃))
            νt[i] = ν
            Pt[i] = P
        end

        νt, Pt, C
    end

    if false
        ff(s,ℙ) = (u) -> Bridge.b(s,u,ℙ)
        s = 1.0
        x = vT
        ff(s,ℙ)(x)
        #B = jacobian(u -> ff(s,ℙ)(u), x)
        B = jacobian(ff(s,ℙ), x)
    end

end


################ plotting in R ############
using RCall
dd = df_iterates

@rput dd
@rput obs_scheme
@rput outdir

R"""
library(ggplot2)
library(tidyverse)
theme_set(theme_bw(base_size = 13))

vT = c(0.03125,   0.25,   1.0)                  #vT <- c(5/128,3/8,2)
vTvec = rep(vT, nrow(dd)/3)

dd$component <- as.factor(dd$component)
dd <- dd %>% mutate(component=fct_recode(component,'component 1'='1','component 2'='2','component 3'='3'))%>% mutate(trueval = vTvec)

# make figure
p <- ggplot(mapping=aes(x=time,y=value,colour=iteration),data=dd) +
  geom_path(aes(group=iteration)) + geom_hline(aes(yintercept=trueval)) +
  facet_wrap(~component,ncol=1,scales='free_y')+
  scale_colour_gradient(low='green',high='blue')+ylab("")
show(p)

# write to pdf
fn <- paste0(outdir,obs_scheme,".pdf")
pdf(fn,width=7,height=5)
show(p)
dev.off()    

"""


