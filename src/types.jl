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
    Observation{Tt, Tv, TL, TΣ, Th}

    (t,v,L,Σ): at time t we have observations v ~ N(Lx_t, Σ)
    h: htransform of the observation 
"""
struct Observation{Tt, Tv, TL, TΣ, Th}
    t::Tt
    v::Tv
    L::TL
    Σ::TΣ
    h::Th
    Observation(t::Tt, v::Tv, L::TL, Σ::TΣ, h::Th) where {Tt,Tv,TL,TΣ,Th} =
        new{Tt, Tv, TL, TΣ, Th}(t,v,L,Σ,h)


    function Observation(t::Tt, v::Tv, L::TL, Σ::TΣ) where {Tt, Tv, TL, TΣ}
        h = Htransform(Obs(), v, L, Σ)
        new{Tt, Tv, TL, TΣ, typeof(h)}(t,v,L,Σ,h)
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
    function Message(::RK4, ℙ, ℙ̃, tt_, hT::Htransform{TH, TF, TC}) where {TH, TF,TC}
        tt = collect(tt_)
        N = length(tt)
        Ht = zeros(TH, N)
        Ft = zeros(TF, N)
        _, _, C = pbridgeode_HFC!(RK4(), ℙ̃, tt, (Ht, Ft), hT)
        new{Bridge.valtype(ℙ), typeof(ℙ), typeof(ℙ̃), eltype(Ht), eltype(Ft), typeof(C)}(ℙ, ℙ̃, tt, Ht, Ft, C)
    end
 
    function Message(D::DE, ℙ, ℙ̃, tt_, hT::Htransform{TH, TF, TC}) where {TH, TF,TC}
        tt = collect(tt_)
        N = length(tt)
        Ht = zeros(TH, N)
        Ft = zeros(TF, N)
        _, _, C = pbridgeode_HFC!(D, ℙ̃, tt, (Ht, Ft), hT)
        new{Bridge.valtype(ℙ), typeof(ℙ), typeof(ℙ̃), eltype(Ht), eltype(Ft), typeof(C)}(ℙ, ℙ̃, tt, Ht, Ft, C)
    end
end


struct Obs end # for dispatch in Htransform
struct Htransform{TH, TF, TC}
    H::TH
    F::TF
    C::TC

    Htransform(H::TH, F::TF, C::TC) where {TH,TF,TC} = new{TH,TF,TC}(H,F,C)
    
    function Htransform(M::Message) 
        new{eltype(M.H), eltype(M.F), typeof(M.C)}(M.H[1], M.F[1], M.C)
    end
    """
        Htransform(v, L, Σ)

        Convert observation v ~ N(Lx, Σ)
        to triplet (H, F, C), which is of type Htransform
    """
    function Htransform(::Obs, v, L, Σ)
        A = L' * inv(Σ)
        H, F, C = A * L, A*v, logpdf(Bridge.Gaussian(zero(v), Σ), v) 
        new{typeof(H), typeof(F), typeof(C)}(H, F, C)
    end
end


mutable struct Chain{TM, TP, THtransform, Tθ}
    Ms::Vector{TM}
    Ps::Vector{TP}
    Msᵒ::Vector{TM}
    Psᵒ::Vector{TP}
    ll::Float64
    h0::THtransform
    θs::Vector{Tθ}
end