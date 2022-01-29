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
  

struct Innovations{T}
    z::Vector{T}
    
   Innovations(z::Vector{T})  where {T} = new{T}(z)
   function Innovations(timegrid, ℙ) 
      z = [sample(timegrids[i], wienertype(ℙ)) for i in eachindex(timegrids) ]  # innovations process
      new{eltype(z)}(z)
   end
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

# struct PathInnovation{TX, TW, Tll}
#     X::TX
#     W::TW
#     ll::Tll
#     PathInnovation(X::TX, W::TW, ll::Tll) where {TX, TW, Tll} = new{TX,TW,Tll}(X, W, ll)

#     function PathInnovation(x0, M)
#         tt = M.tt
#         W = sample(tt, wienertype(M.ℙ))    
#         X = solve(Euler(), x0, W, M)  # allocation        
#         ll = llikelihood(Bridge.LeftRule(), X, M, skip=sk)
#         new{typeof(X), typeof(W), typeof(ll)}(X, W, ll)
#     end
# end



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
struct Message{Tℙ̃,TH,TF,TC} 
    ℙ̃::Tℙ̃   
    tt::Vector{Float64}  
    H::Vector{TH}      
    F::Vector{TF}      
    C::TC              
    Message(ℙ̃::Tℙ̃, tt, Ht::Vector{TH}, Ft::Vector{TF}, C::TC) where {Tℙ̃,TH,TF,TC} =
        new{Tℙ̃,TH,TF,TC}(ℙ̃, tt, Ht, Ft, C)

    function Message(D::DE, ℙ̃, tt_, hT::Htransform{TH, TF, TC}) where {TH, TF,TC}
        tt = collect(tt_)
        N = length(tt)
        Ht = zeros(TH, N)
        Ft = zeros(TF, N)
        _, _, C = pbridgeode_HFC!(D, ℙ̃, tt, (Ht, Ft), hT)
        new{typeof(ℙ̃), eltype(Ht), eltype(Ft), typeof(C)}(ℙ̃, tt, Ht, Ft, C)
    end
end


function Htransform(M::Message) 
    Htransform(M.H[1], M.F[1], M.C)
end


struct BackwardFilter{T, Th0}
    Ms::Vector{T}
    h0::Th0
    
   BackwardFilter(Ms, h0::Th0)  where {Th0} = new{eltype(Ms), Th0}(Ms, h0)
   
   function BackwardFilter(AuxType, obs, timegrids, x0, guidingterm_with_x1) 
        h0, Ms = init_auxiliary_processes(AuxType, obs, timegrids, x0, guidingterm_with_x1)
        new{eltype(Ms), typeof(h0)}(Ms, h0)
    end
end  
