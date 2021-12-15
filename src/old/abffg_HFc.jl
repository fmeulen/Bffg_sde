# function kernelr3(f, t, y, dt, P)
#     k1 = f(t, y, P)
#     k2 = f(t + 1/2*dt, y + 1/2*dt*k1, P)
#     k3 = f(t + 3/4*dt, y + 3/4*dt*k2, P)
#     y + dt*(2/9*k1 + 1/3*k2 + 4/9*k3)
# end


function kernelrk4(f, t, y, dt, P)
    k1 = f(t, y, P)
    k2 = f(t + 0.5*dt, y + 0.5*k1*dt, P)
    k3 = f(t + 0.5*dt, y + 0.5*k2*dt, P)
    k4 = f(t + dt, y + k3*dt, P)
    y + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
end



cd("/Users/frankvandermeulen/Sync/DOCUMENTS/onderzoek/code/diffbridges")
outdir="/Users/frankvandermeulen/Sync/DOCUMENTS/onderzoek/code/diffbridges/out_nclar/"


using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
using ForwardDiff

import Bridge: R3, IndexedTime, llikelihood, kernelr3, constdiff
import ForwardDiff: jacobian

struct PBridge{T,TP,TP̃,TF,TH,TC} <: ContinuousTimeProcess{T}
    P::TP   # diffusion 
    P̃::TP̃   # auxiliary process
    tt::Vector{Float64}  # time grid
    F::Vector{TF}        # F values on time grid
    H::Vector{TH}        # H values on time grid
    C::TC                # constant to compute h-function
    PBridge(P::TP, P̃::TP̃, tt, Ft::Vector{TF}, Ht::Vector{TH}, C::TC) where {TP,TP̃,TF,TH,TC} =
        new{Bridge.valtype(P),TP,TP̃,TF,TH,TC}(P, P̃, tt, Ft, Ht, C)

    # constructor: provide (timegrid, P, P̃, FT, HT, CT)    
    function PBridge(P, P̃, tt_, FT::TF, HT::TH, CT) where {TF, TH}
        tt = collect(tt_)
        N = length(tt)
        Ht = zeros(TH, N)
        Ft = zeros(TF, N)
        _, _, C = pbridgeode!(R3(), P̃, tt, (Ft, Ht), (FT, HT, CT))
        PBridge(P, P̃, tt, Ft, Ht, C)
    end

    # 2nd constructor (use also samplepath X in backward ODEs): provide (timegrid, P, FT, HT, CT, X::SamplePath)    
    function PBridge(P, P̃, tt_, FT::TF, HT::TH, CT, X::SamplePath) where {TF, TH}
        tt = collect(tt_)
        N = length(tt)
        Ht = zeros(TH, N)
        Ft = zeros(TF, N)
        _, _, C = pbridgeode!(R3(), P, tt, (Ft, Ht), (FT, HT, CT), X)
        PBridge(P, P̃, tt, Ft, Ht, C)
    end
end


function pbridgeode!(::R3, P̃, t, (Ft, Ht), (FT, HT, CT))
    Ht[end] = HT
    Ft[end] = FT
    F, H, C = FT, HT, CT

    dH(s, y, P̃) = - Bridge.B(s, P̃)'*y - y * Bridge.B(s,P̃) + y*Bridge.a(s, P̃)*y'
    dF(s, y, (H,P̃)) = -Bridge.B(s, P̃)'*y + H*Bridge.a(s, P̃)*y  + H*Bridge.β(s, P̃)

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        C += ( Bridge.β(t[i+1], P̃)'*F + 0.5*F'*Bridge.a(t[i+1], P̃)*F - 0.5*tr(H*Bridge.a(t[i+1], P̃)) ) * dt
        H = kernelrk4(dH, t[i+1], H, dt, P̃)
        F = kernelrk4(dF, t[i+1], F, dt, (H, P̃))
        Ft[i] = F
        Ht[i] = H
    end

    Ft, Ht, C
end

H((i,t)::IndexedTime, x, 𝒫::PBridge) = 𝒫.H[i]
F((i,t)::IndexedTime, x, 𝒫::PBridge) = 𝒫.F[i]
r((i,t)::IndexedTime, x, 𝒫::PBridge) = 𝒫.F[i] - 𝒫.H[i]*x

function Bridge._b((i,t)::IndexedTime, x, 𝒫::PBridge)  
    Bridge.b(t, x, 𝒫.P) + Bridge.a(t, x, 𝒫.P) * r((i,t),x,𝒫)   # (𝒫.F[i] - 𝒫.H[i]*x)
end


Bridge.σ(t, x, 𝒫::PBridge) = Bridge.σ(t, x, 𝒫.P)
Bridge.a(t, x, 𝒫::PBridge) = Bridge.a(t, x, 𝒫.P)
Bridge.Γ(t, x, 𝒫::PBridge) = Bridge.Γ(t, x, 𝒫.P)
Bridge.constdiff(𝒫::PBridge) = Bridge.constdiff(𝒫.P) && Bridge.constdiff(𝒫.P̃)

logh̃(x, 𝒫::PBridge) =  -0.5 * x' * 𝒫.H[1] * x + 𝒫.F[1]' * x - 𝒫.C

function llikelihood(::LeftRule, X::SamplePath, 𝒫::PBridge; skip = 0)
    tt = X.tt
    xx = X.yy

    som::Float64 = 0.
    for i in 1:length(tt)-1-skip #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        r̃ = r((i,s), x, 𝒫)
        dt = tt[i+1]-tt[i]

        som += dot( Bridge._b((i,s), x, 𝒫.P) - Bridge._b((i,s), x, 𝒫.P̃), r̃) 
        if !constdiff(𝒫)
            H = H((i,s), x, 𝒫)
            som -= 0.5*tr( (a((i,s), x, 𝒫.P) - a((i,s), x, 𝒫.P̃)) * H )   
            som += 0.5 *( r̃'* ( a((i,s), x, 𝒫.P) - a((i,s), x, 𝒫.P̃) ) * r̃ ) 
        end
        som *= dt 
    end
    som #+ logh̃(X.yy[1], 𝒫)
end

function initFHC(PT_, vT, LT)
    d,  = size(PT_)
    z = SVector{d}(zeros(d))
    HT_ = inv(PT_)
    CT_ = -logpdf(Bridge.Gaussian(z, PT_), z) 
    FT_ = PT_ \ (LT\vT)
    FT_, HT_, CT_
end



function fusion(FT_, HT_, CT_, vT, LT, ΣT)
    d = size(ΣT)[1]
    A = LT' * inv(ΣT)
    FT_ + A*vT, HT_ + A*LT, CT_ - logpdf(Bridge.Gaussian(0.0*vT, ΣT), vT)
end



################################################################################



# solve ODEs using current SamplePath X (need to pass P for that, instead of P̃)
function pbridgeode!(::R3, P, t, (Ft, Ht), (FT, HT, CT), X)
    Ht[end] = HT
    Ft[end] = FT
    F, H, C = FT, HT, CT


    # function dH(s, y, (P,x)) 
    #     #ff(s,P) = u -> Bridge.b(s,u,P)
    #     #B = jacobian(u->ff(s,P)(u), x)
    #     B = jacobianb(s,x,P)
    #     out = - B'*y - y*B + y*Bridge.a(s, x, P)*y'
    #     return out
    #  end
    
    dH(s, y, (B̃, ã)) = - B̃'*y - y * B̃ + y*ã*y'


    #dF(s, y, (H,P̃)) = -Bridge.B(s, P̃)'*y + H*Bridge.a(s, P̃)*y  + H*Bridge.β(s, P̃)
    # function dF(s, y, (H,P,x)) 
    #     #B = jacobian(u->Bridge.b(s,u,P),x)
    #     B = jacobianb(s,x,P)
    #     out = -B'*y + H*Bridge.a(s, x, P)*y + H*(Bridge.b(s,x,P) - B*x)
    #     return out
    #  end

     dF(s,y, (B̃, ã, β̃)) = -B̃'*y + H*ã*y + H*β̃

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        x = X.yy[i+1]
        s = t[i+1]

        B̃ = jacobianb(s,x,P)
        β̃ = Bridge.b(s,x,P) - B̃*x
        ã =  Bridge.a(s, x, P)

        C += ( β̃'*F + 0.5*F'*ã*F - 0.5*tr(H*ã) ) * dt
        #H = kernelr3(dH, t[i+1], H, dt, (P,x))
        H = kernelrk4(dH, t[i+1], H, dt, (B̃, ã))
        #F = kernelr3(dF, t[i+1], F, dt, (H, P, x))
        F = kernelrk4(dF, t[i+1], F, dt, (B̃, ã, β̃))
        Ft[i] = F
        Ht[i] = H
    end

    Ft, Ht, C
end

if false
    ff(s,P) = (u) -> Bridge.b(s,u,P)
    s = 1.0
    x = vT
    ff(s,P)(x)
    #B = jacobian(u -> ff(s,P)(u), x)
    B = jacobian(ff(s,P), x)
end





