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
    PathInnovation(X::TX, W::TW, ll::Tll) where {TX, TW, Tll} = new{TX,TW,Tll}(X, W, ll)

    function PathInnovation(x0, M)
        tt = M.tt
        W = sample(tt, wienertype(M.ℙ))    
        X = solve(Euler(), x0, W, M)  # allocation        
        ll = llikelihood(Bridge.LeftRule(), X, M, skip=sk)
        new{typeof(X), typeof(W), typeof(ll)}(X, W, ll)
    end
end


struct PathInnovationProposal{TX, TW, Tll}
    X::TX
    W::TW
    ll::Tll
    Wbuf::TW
    ρ::Float64
    PathInnovationProposal(X::TX, W::TW, ll::Tll, Wbuf::TW, ρ::Float64) where {TX, Tll, TW} =
    new{TX,TW,Tll}(X, W, ll, Wbuf, ρ)

    function PathInnovationProposal(P::PathInnovation, ρ)
       Wbuf = deepcopy(P.W)
       W = deepcopy(P.W)
       X = deepcopy(P.X)
       ll = deepcopy(P.ll)
       new{typeof(X), typeof(W), typeof(ll)}(X, W, ll, Wbuf, ρ)
    end
end

PathInnovation(P::PathInnovationProposal) = PathInnovation(copy(P.X), copy(P.W), P.ll)


struct Obs end # for dispatch in Htransform
struct Htransform{TH, TF, TC}
    H::TH
    F::TF
    C::TC

    Htransform(H::TH, F::TF, C::TC) where {TH,TF,TC} = new{TH,TF,TC}(H,F,C)
    
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


function Htransform(M::Message) 
    Htransform(M.H[1], M.F[1], M.C)
end


struct ChainState{TM, TP, TPᵒ, THtransform, Tθ}
    Ms::Vector{TM}
    Ps::Vector{TP}
    Msᵒ::Vector{TM}
    Psᵒ::Vector{TPᵒ}
    ll::Float64
    h0::THtransform
    θ::Tθ

    ChainState(Ms, Ps, Msᵒ, Psᵒ, ll, h0, θ)  = new{eltype(Ms),eltype(Ps), eltype(Psᵒ), typeof(h0), typeof(θ)}(Ms, Ps, Msᵒ, Psᵒ, ll, h0, θ)
    

    function ChainState(ρ::Float64, timegrids, obs, ℙ, x0, pars, guidingterm_with_x1; AuxType=JansenRitDiffusionAux)
        ℙ̃s = AuxType[]
        n = length(obs)
        for i in 2:n # skip x0
          lininterp = LinearInterpolation([obs[i-1].t,obs[i].t], zeros(2) )
          push!(ℙ̃s, AuxType(obs[i].t, obs[i].v[1], lininterp, false, ℙ))
        end
        h0, Ms = backwardfiltering(obs, timegrids, ℙ, ℙ̃s)
        if guidingterm_with_x1
            add_deterministicsolution_x1!(Ms, x0)
            h0 = backwardfiltering!(Ms, obs)
        end
        
        ρs = fill(ρ, length(timegrids))    
        Ps = forwardguide(x0, Ms);
        ll = loglik(x0, h0, Ps)
        θ = getpar(Ms, pars)
        Psᵒ = [PathInnovationProposal(Ps[i], ρs[i]) for i ∈ eachindex(Ps)] 
        #Psᵒ[2].W.yy === Ps[2].W.yy 
        Msᵒ = deepcopy(Ms)
        new{eltype(Ms), eltype(Ps),  eltype(Psᵒ), typeof(h0), typeof(θ)}(Ms, Ps, Msᵒ, Psᵒ, ll, h0, θ)
    end
end

# import Base.show
# Base.show(io::IO, S::ChainState)
#     print(io, S.Ms)
#     println()
#     print(io, S.θ)
# end



# struct CS{Tℙ, Tℙ̃, TX, TW, TM, Tl, Tl0, THtransform, Tθ}
#     ℙ::Tℙ
#     ℙ̃::Vector{Tℙ̃}   
#     XX::Vector{TX}
#     WW::Vector{TW}
#     MM::Vector{TM}
#     ll::Vector{Tl}
#     loglik::Tl0
#     h0::THtransform
#     θ::Tθ
    
#     CS(ℙ, ℙ̃, XX, WW, MM, ll, loglik, h0, θ)  = new{typeof(ℙ), eltype(ℙ̃), eltype(XX),eltype(WW), eltype(MM), eltype(ll), typeof(loglik), typeof(h0), typeof(θ)}(ℙ, ℙ̃, XX, WW, MM, ll, loglik, h0, θ)
    
#     function ChainState(ρ::Float64, timegrids, obs, ℙ, x0, pars, guidingterm_with_x1; AuxType=JansenRitDiffusionAux)
#         ℙ̃ = AuxType[]
#         n = length(obs)
#         for i in 2:n # skip x0
#           lininterp = LinearInterpolation([obs[i-1].t,obs[i].t], zeros(2) )
#           push!(ℙ̃, AuxType(obs[i].t, obs[i].v[1], lininterp, false, ℙ))
#         end
#         h0, MM = backwardfiltering(obs, timegrids, ℙ, ℙ̃)
#         if guidingterm_with_x1
#             add_deterministicsolution_x1!(MM, x0)
#             h0 = backwardfiltering!(MM, obs)
#         end
        
#         ρs = fill(ρ, length(timegrids))    
#         xend = x0
#         for i in eachindex(timegrids)
#             tt = timegrids[i]
#             W = sample(tt, wienertype(ℙ))    
#             X = solve(Euler(), xend, W, M)  # allocation        
#         ll = llikelihood(Bridge.LeftRule(), X, M, skip=sk)


#         Ps = forwardguide(x0, Ms);
#         ll = loglik(x0, h0, Ps)
#         θ = getpar(Ms, pars)
#         Psᵒ = [PathInnovationProposal(Ps[i], ρs[i]) for i ∈ eachindex(Ps)] 
#         #Psᵒ[2].W.yy === Ps[2].W.yy 
#         Msᵒ = deepcopy(Ms)
#         new{eltype(Ms), eltype(Ps),  eltype(Psᵒ), typeof(h0), typeof(θ)}(Ms, Ps, Msᵒ, Psᵒ, ll, h0, θ)
#     end
# end