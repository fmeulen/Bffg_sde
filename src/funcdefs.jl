"""
    kernelrk4(f, t, y, dt, ℙ)

    solver for Runge-Kutta 4 scheme
"""
function kernelrk4(f, t, y, dt, ℙ)
    k1 = f(t, y, ℙ)
    k2 = f(t + 0.5*dt, y + 0.5*k1*dt, ℙ)
    k3 = f(t + 0.5*dt, y + 0.5*k2*dt, ℙ)
    k4 = f(t + dt, y + k3*dt, ℙ)
    y + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
end

"""
    convert_HFC_to_PνC(H,F,C)

    convert parametrisation 
        exp(-c - 0.5 x' H x + F' x)
    to 
        exp(-c - 0.5 x'P^(-1) x + (P^(-1) nu)' x)
"""
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

implemented by M. Mider in GuidedProposals.jl
"""
function static_accessor_HFc(u::K, ::Val{T}) where {K<:Union{SVector,MVector},T}
    Hidx = SVector{T*T,Int64}(1:T*T)
    Fidx = SVector{T,Int64}((T*T+1):(T*T+T))
    reshape(u[Hidx], Size(T,T)), u[Fidx], u[T*T+T+1]
end
"""
    PBridge

        struct for partial bridges
    ℙ:  target diffusion
    ℙ̃:  auxiliary NclarDiffusion
    tt: time grid for diffusion (including start and end time)
    P:  P-values on tt
    ν:  ν values on tt
    C:  -C is an additive factor in the loglikelihood

    constructor from incomsing triplet (PT, νT, cT) is given by 
        PBridge(ℙ, ℙ̃, tt, PT, νT, CT) 
"""
struct PBridge{T,Tℙ,Tℙ̃,TP,Tν,TC} <: ContinuousTimeProcess{T}
    ℙ::Tℙ   
    ℙ̃::Tℙ̃   
    tt::Vector{Float64}  
    P::Vector{TP}      
    ν::Vector{Tν}      
    C::TC              
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

"""
    pbridgeode!(::R3, ℙ̃, t, (Pt, νt), (PT, νT, CT))

    Solve backward ODEs for `(P, ν, C)` starting from `(PT, νT, CT)`` on time grid `t``
    Auxiliary process is given by ℙ̃
    Writes into (Pt, νt)
"""
function pbridgeode!(::R3, ℙ̃, t, (Pt, νt), (PT, νT, CT))
    

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
    access = Val{}(d)
    y = vectorise(PT, νT, CT)

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        y = kernelrk4(dPνC, t[i+1], y, dt, ℙ̃)
        Pt[i], νt[i], C = static_accessor_HFc(y, access)
    end

    Pt, νt, C
end

"""
    init_HFC(v, L; ϵ=0.01)

    First computes xT = L^(-1) * vT (Moore-Penrose inverse), a reasonable guess for the full state based on the partial observation vT
    Then convert artifical observation v ~ N(xT, ϵ^(-1) * I)
    to triplet  (H, F, C)
"""
function init_HFC(v, L; ϵ=0.01)
    P = ϵ^(-1)*SMatrix{d,d}(1.0I)
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



########### specify pars for 𝒫::PBridge
# P((i,t)::IndexedTime, 𝒫::PBridge) = 𝒫.P[i]
# ν((i,t)::IndexedTime, 𝒫::PBridge) = 𝒫.ν[i]

r((i,t)::IndexedTime, x, 𝒫::PBridge) = (𝒫.P[i] \ (𝒫.ν[i] - x) )

function Bridge._b((i,t)::IndexedTime, x, 𝒫::PBridge)  
    Bridge.b(t, x, 𝒫.ℙ) + Bridge.a(t, x, 𝒫.ℙ) * r((i,t),x,𝒫)   
end

Bridge.σ(t, x, 𝒫::PBridge) = Bridge.σ(t, x, 𝒫.ℙ)
Bridge.a(t, x, 𝒫::PBridge) = Bridge.a(t, x, 𝒫.ℙ)
#Bridge.Γ(t, x, 𝒫::PBridge) = Bridge.Γ(t, x, 𝒫.ℙ)
Bridge.constdiff(𝒫::PBridge) = Bridge.constdiff(𝒫.ℙ) && Bridge.constdiff(𝒫.ℙ̃)

function logh̃(x, 𝒫::PBridge) 
    H1, F1, C = convert_PνC_to_HFC(𝒫.P[1], 𝒫.ν[1],𝒫.C)
    -0.5 * x' * H1 * x + F1' * x - C    
end

function llikelihood(::LeftRule, X::SamplePath, 𝒫::PBridge; skip = 0, include_h0=false)
    tt = X.tt
    xx = X.yy

    som::Float64 = 0.
    for i in 1:length(tt)-1-skip #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        r̃ = r((i,s), x, 𝒫)
        dt = tt[i+1]-tt[i]

        som += dot( Bridge._b((i,s), x, 𝒫.ℙ) - Bridge._b((i,s), x, 𝒫.ℙ̃), r̃) * dt
        if !constdiff(𝒫)
            P = 𝒫.P[i]  #P((i,s), x, 𝒫)
            som -= 0.5*tr( (a((i,s), x, 𝒫.ℙ) - a((i,s), x, 𝒫.ℙ̃)) * P )   * dt
            som += 0.5 *( r̃'* ( a((i,s), x, 𝒫.ℙ) - a((i,s), x, 𝒫.ℙ̃) ) * r̃ ) * dt
        end
    end

    som + (include_h0) * logh̃(X.yy[1], 𝒫)
end




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


