abstract type Solver end

struct RK4 <: Solver end
struct DE{T} <: Solver 
    solvertype::T
end

struct Adaptive <: Solver end
    struct AssumedDensityFiltering{T} <: Solver 
    solvertype::T
end


abstract type  GuidType end
struct PCN <: GuidType  end
struct InnovationsFixed <: GuidType end

struct ParInfo
    names::Vector{Symbol}
    recomputeguidingterm::Vector{Bool}
end
  

"""
    Observation{Tt, Tv, TL, TΣ, TH, TF, TC}

    (t,v,L,Σ): at time t we have observations v ~ N(Lx_t, Σ)
    (H, F, C): message from the observation to the triple`
"""
struct Observation{Tt, Tv, TL, TΣ, TH, TF, TC}
    t::Tt
    v::Tv
    L::TL
    Σ::TΣ
    H::TH
    F::TF
    C::TC
    Observation(t::Tt, v::Tv, L::TL, Σ::TΣ, H::TH, F::TF, C::TC) where {Tt,Tv,TL,TΣ,TH, TF, TC} =
        new{Tt, Tv, TL, TΣ, TH, TF, TC}(t,v,L,Σ,H,F,C)


    function Observation(t::Tt, v::Tv, L::TL, Σ::TΣ) where {Tt, Tv, TL, TΣ}
        H, F, C = observation_HFC(v, L, Σ)
        new{Tt, Tv, TL, TΣ, typeof(H), typeof(F), typeof(C)}(t,v,L,Σ,H,F,C)
    end    
end

"""
    PathInnovation{TX, TW, Tll}

    contains path, innovation and loglikelihood for a segment (=kernel)
    additionally contains buffers for proposals to be used in a pCN step
"""
struct PathInnovation{TX, TW, Tll}
    X::TX
    W::TW
    ll::Tll
    Wbuf::TW
    ρ::Float64
    PathInnovation(X::TX, W::TW, ll::Tll, Wbuf::TW, ρ::Float64) where {TX, Tll, TW} =
    new{TX,TW,Tll}(X, W, ll, Wbuf, ρ)

    function PathInnovation(x0, M, ρ)
        tt = M.tt
        W = sample(tt, wienertype(M.ℙ))    
        X = solve(Euler(), x0, W, M)  # allocation        
        ll = llikelihood(Bridge.LeftRule(), X, M, skip=sk)
        Wbuf = deepcopy(W)
        new{typeof(X), typeof(W), typeof(ll)}(X, W, ll, Wbuf, ρ)
    end
end


"""
    Message

        struct containing all information for guiding on a segment (equivalently kernel)
    ℙ:  target diffusion
    ℙ̃:  auxiliary NclarDiffusion
    tt: time grid for diffusion (including start and end time)
    H:  H-values on tt
    F:  F values on tt
    C:  C is an additive factor in the loglikelihood

    constructors for solving the backward filtering by numerically approximating ODEs
"""
struct Message{T,Tℙ,Tℙ̃,TH,TF,TC} <: ContinuousTimeProcess{T}
    ℙ::Tℙ   
    ℙ̃::Tℙ̃   
    tt::Vector{Float64}  
    H::Vector{TH}      
    F::Vector{TF}      
    C::TC              
    Message(ℙ::Tℙ, ℙ̃::Tℙ̃, tt, Ht::Vector{TH}, Ft::Vector{TF}, C::TC) where {Tℙ,Tℙ̃,TH,TF,TC} =
        new{Bridge.valtype(ℙ),Tℙ,Tℙ̃,TH,TF,TC}(ℙ, ℙ̃, tt, Ht, Ft, C)

    # constructor: provide (ℙ, ℙ̃, timegrid HT, FT, CT)    
    function Message(::RK4, ℙ, ℙ̃, tt_, HT::TH, FT::TF, CT) where {TH, TF}
        tt = collect(tt_)
        N = length(tt)
        Ht = zeros(TH, N)
        Ft = zeros(TF, N)
        _, _, C = pbridgeode_HFC!(RK4(), ℙ̃, tt, (Ht, Ft), (HT, FT, CT))
        Message(ℙ, ℙ̃, tt, Ht, Ft, C)
    end
 
    function Message(D::DE, ℙ, ℙ̃, tt_, HT::TH, FT::TF, CT) where {TH, TF}
        tt = collect(tt_)
        N = length(tt)
        Ht = zeros(TH, N)
        Ft = zeros(TF, N)
        _, _, C = pbridgeode_HFC!(D, ℙ̃, tt, (Ht, Ft), (HT, FT, CT))
        new{eltype(Ft), typeof(ℙ), typeof(ℙ̃), eltype(Ht), eltype(Ft), typeof(C)}(ℙ, ℙ̃, tt, Ht, Ft, C)
    end
end

mutable struct Chain
    Ms::Vector{Message}
    Ps::Vector{PathInnovation}
    Msᵒ::Vector{Message}
    Psᵒ::Vector{PathInnovation}
    loglik::Float64
    HFC0::Tuple
    θs::Vector
end