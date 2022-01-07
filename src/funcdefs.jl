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

HFC(obs::Observation) = (obs.H, obs.F, obs.C)

# FIXME timechange grid
timegrid(t0, t1; M) = collect(range(t0, t1, length=M))



struct PathInnovation{TX, TW, Tll}
    X::TX
    W::TW
    ll::Tll
    Xᵒ::TX
    Wᵒ::TW
    Wbuf::TW
    PathInnovation(X::TX, W::TW, ll::Tll, Xᵒ::TX, Wᵒ::TW, Wbuf::TW) where {TX, Tll, TW} =
    new{TX,TW,Tll}(X, W, ll, Xᵒ, Wᵒ, Wbuf)

    function PathInnovation(x0, 𝒫)
        tt = 𝒫.tt
        W = sample(tt, wienertype(𝒫.ℙ))    #W = sample(tt, Wiener())
        X = solve(Euler(), x0, W, 𝒫.ℙ)  # allocation
        solve!(Euler(),X, x0, W, 𝒫)
        Xᵒ = deepcopy(X)
        ll = llikelihood(Bridge.LeftRule(), X, 𝒫, skip=sk)
        Wᵒ = deepcopy(W)
        Wbuf = deepcopy(W)
        TX, TW, Tll = typeof(X), typeof(W), typeof(ll)
        new{TX, TW, Tll}(X,W,ll,Xᵒ, Wᵒ, Wbuf)
    end
end



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


abstract type Solver end

struct RK4 <: Solver end
struct DE{T} <: Solver 
    solvertype::T
end

struct Adaptive <: Solver end
    
struct AssumedDensityFiltering{T} <: Solver 
    solvertype::T
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
    function PBridge(::RK4, ℙ, ℙ̃, tt_, PT::TP, νT::Tν, CT) where {TP, Tν}
        tt = collect(tt_)
        N = length(tt)
        Pt = zeros(TP, N)
        νt = zeros(Tν, N)
        _, _, C = pbridgeode!(RK4(), ℙ̃, tt, (Pt, νt), (PT, νT, CT))
        PBridge(ℙ, ℙ̃, tt, Pt, νt, C)
    end

    
    function PBridge(D::DE, ℙ, ℙ̃, tt_, PT::TP, νT::Tν, CT) where {TP, Tν}
        tt = collect(tt_)
        N = length(tt)
        Pt = zeros(TP, N)
        νt = zeros(Tν, N)
        _, _, C = pbridgeode!(D, ℙ̃, tt, (Pt, νt), (PT, νT, CT))
        PBridge(ℙ, ℙ̃, tt, Pt, νt, C)
    end

    function PBridge(D::AssumedDensityFiltering, ℙ,  ℙ̃, tt_, PT::TP, νT::Tν, CT) where {TP, Tν}
        tt = collect(tt_)
        N = length(tt)
        Pt = zeros(TP, N)
        νt = zeros(Tν, N)
        _, _, C = pbridgeode!(D, ℙ, tt, (Pt, νt), (PT, νT, CT))
        PBridge(ℙ, ℙ̃, tt, Pt, νt, C)
    end



    # 2nd constructor (use also samplepath X in backward ODEs): provide (timegrid, ℙ, νT, PT, CT, X::Sampleℙath)    
    function PBridge(::Adaptive, ℙ, ℙ̃, tt_, PT::TP, νT::Tν, CT, X::SamplePath) where  {TP, Tν}
        tt = collect(tt_)
        N = length(tt)
        Pt = zeros(TP, N)
        νt = zeros(Tν, N)
        _, _, C = pbridgeode!(Adaptive(), ℙ, tt, (Pt, νt), (PT, νT, CT), X)
        PBridge(ℙ, ℙ̃, tt, Pt, νt, C)
    end
end

"""
    pbridgeode!(::RK4, ℙ̃, t, (Pt, νt), (PT, νT, CT))

    Solve backward ODEs for `(P, ν, C)` starting from `(PT, νT, CT)`` on time grid `t``
    Auxiliary process is given by ℙ̃
    Writes into (Pt, νt)
"""
function pbridgeode!(::RK4, ℙ̃, t, (Pt, νt), (PT, νT, CT))
    

    function dPνC(s, y, ℙ̃)
        access = Val{}(dim(ℙ̃))
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
    access = Val{}(dim(ℙ̃))
    y = vectorise(PT, νT, CT)

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        y = kernelrk4(dPνC, t[i+1], y, dt, ℙ̃)
        Pt[i], νt[i], C = static_accessor_HFc(y, access)
    end

    Pt, νt, C
end

#---------- also for HFC --------------------
"""
    GuidedProcess

        struct for partial bridges
    ℙ:  target diffusion
    ℙ̃:  auxiliary NclarDiffusion
    tt: time grid for diffusion (including start and end time)
    H:  H-values on tt
    F:  F values on tt
    C:  -C is an additive factor in the loglikelihood

    constructor from incomsing triplet (PT, νT, cT) is given by 
        GuidedProcess(ℙ, ℙ̃, tt, PT, νT, CT) 
"""
struct GuidedProcess{T,Tℙ,Tℙ̃,TP,Tν,TC} <: ContinuousTimeProcess{T}
    ℙ::Tℙ   
    ℙ̃::Tℙ̃   
    tt::Vector{Float64}  
    H::Vector{TP}      
    F::Vector{Tν}      
    C::TC              
    GuidedProcess(ℙ::Tℙ, ℙ̃::Tℙ̃, tt, Ht::Vector{TP}, Ft::Vector{Tν}, C::TC) where {Tℙ,Tℙ̃,TP,Tν,TC} =
        new{Bridge.valtype(ℙ),Tℙ,Tℙ̃,TP,Tν,TC}(ℙ, ℙ̃, tt, Ht, Ft, C)

    # constructor: provide (timegrid, ℙ, ℙ̃, νT, PT, CT)    
    function GuidedProcess(::RK4, ℙ, ℙ̃, tt_, HT::TP, FT::Tν, CT) where {TP, Tν}
        tt = collect(tt_)
        N = length(tt)
        Ht = zeros(TP, N)
        Ft = zeros(Tν, N)
        _, _, C = pbridgeode_HFC!(RK4(), ℙ̃, tt, (Ht, Ft), (HT, FT, CT))
        GuidedProcess(ℙ, ℙ̃, tt, Ht, Ft, C)
    end

    
    function GuidedProcess(D::DE, ℙ, ℙ̃, tt_, PT::TP, νT::Tν, CT) where {TP, Tν}
        tt = collect(tt_)
        N = length(tt)
        Pt = zeros(TP, N)
        νt = zeros(Tν, N)
        _, _, C = pbridgeode_HFC!(D, ℙ̃, tt, (Pt, νt), (PT, νT, CT))
        GuidedProcess(ℙ, ℙ̃, tt, Pt, νt, C)
    end


    # # 2nd constructor (use also samplepath X in backward ODEs): provide (timegrid, ℙ, νT, PT, CT, X::Sampleℙath)    
    # function GuidedProcess(::Adaptive, ℙ, ℙ̃, tt_, PT::TP, νT::Tν, CT, X::SamplePath) where  {TP, Tν}
    #     tt = collect(tt_)
    #     N = length(tt)
    #     Pt = zeros(TP, N)
    #     νt = zeros(Tν, N)
    #     _, _, C = pbridgeode_HFC!(Adaptive(), ℙ, tt, (Pt, νt), (PT, νT, CT), X)
    #     PBridge(ℙ, ℙ̃, tt, Pt, νt, C)
    # end
end

"""
    pbridgeode_HFC!(::RK4, ℙ̃, t, (Ht, Ft), (HT, FT, CT))

    Solve backward ODEs for `(H, F, C)` starting from `(HT, FT, CT)`` on time grid `t``
    Auxiliary process is given by ℙ̃
    Writes into (Ht, Ft)
"""
function pbridgeode_HFC!(::RK4, ℙ̃, t, (Ht, Ft), (HT, FT, CT))
    

    function dHFC(s, y, ℙ̃)
        access = Val{}(dim(ℙ̃))
        H, F, _ = static_accessor_HFc(y, access)
        _B, _β, _σ, _a = Bridge.B(s, ℙ̃), Bridge.β(s, ℙ̃), Bridge.σ(s, ℙ̃), Bridge.a(s, ℙ̃)

        dH = - (_B' * H)  - (H * _B) + Bridge.outer( H * _σ)
        dF = - (_B' * F) + H * (_a * F + _β) 
        dC = dot(_β, F) + 0.5*Bridge.outer(F' * _σ) - 0.5*tr( (H* (_a)))
        vectorise(dH, dF, dC)
    end

    Ht[end] = HT
    Ft[end] = FT
    C = CT
    access = Val{}(dim(ℙ̃))
    y = vectorise(HT, FT, CT)

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        y = kernelrk4(dHFC, t[i+1], y, dt, ℙ̃)
        Ht[i], Ft[i], C = static_accessor_HFc(y, access)
    end

    Ht, Ft, C
end




#---------------------------------------------





"""
    init_HFC(v, L, d; ϵ=0.01)

    d = dimension of the diffusion
    First computes xT = L^(-1) * vT (Moore-Penrose inverse), a reasonable guess for the full state based on the partial observation vT
    Then convert artifical observation v ~ N(xT, ϵ^(-1) * I)
    to triplet  (H, F, C)
"""
function init_HFC(v, L, d::Int64; ϵ=0.01)
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

function Bridge._b((i,t)::IndexedTime, x, 𝒫::Union{GuidedProcess, PBridge})  
    Bridge.b(t, x, 𝒫.ℙ) + Bridge.a(t, x, 𝒫.ℙ) * r((i,t),x,𝒫)   
end

Bridge.σ(t, x, 𝒫::Union{GuidedProcess, PBridge}) = Bridge.σ(t, x, 𝒫.ℙ)
Bridge.a(t, x, 𝒫::Union{GuidedProcess, PBridge}) = Bridge.a(t, x, 𝒫.ℙ)
#Bridge.Γ(t, x, 𝒫::PBridge) = Bridge.Γ(t, x, 𝒫.ℙ)
Bridge.constdiff(𝒫::Union{GuidedProcess, PBridge}) = Bridge.constdiff(𝒫.ℙ) && Bridge.constdiff(𝒫.ℙ̃)

function logh̃(x, 𝒫::PBridge) 
    H1, F1, C = convert_PνC_to_HFC(𝒫.P[1], 𝒫.ν[1], 𝒫.C)
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
            som -= 0.5*tr( (a((i,s), x, 𝒫.ℙ) - a((i,s), x, 𝒫.ℙ̃)) / P )   * dt
            som += 0.5 * ( r̃' * ( a((i,s), x, 𝒫.ℙ) - a((i,s), x, 𝒫.ℙ̃) ) * r̃)  * dt
        end
    end

    som + (include_h0) * logh̃(X.yy[1], 𝒫)
end


#------ also for HFC   GuidedProcess
r((i,t)::IndexedTime, x, 𝒫::GuidedProcess) = 𝒫.F[i] - 𝒫.H[i] * x 

function logh̃(x, 𝒫::GuidedProcess) 
    -0.5 * x' * 𝒫.H[1] * x + 𝒫.F[1]' * x - 𝒫.C    
end

function llikelihood(::LeftRule, X::SamplePath, 𝒫::GuidedProcess; skip = 0, include_h0=false)
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
            som -= 0.5*tr( (a((i,s), x, 𝒫.ℙ) - a((i,s), x, 𝒫.ℙ̃)) * 𝒫.H[i] )   * dt
            som += 0.5 * ( r̃' * ( a((i,s), x, 𝒫.ℙ) - a((i,s), x, 𝒫.ℙ̃) ) * r̃)  * dt
        end
    end

    som + (include_h0) * logh̃(X.yy[1], 𝒫)
end




################################################################################

function pbridgeode!(D::DE, ℙ̃, t, (Pt, νt), (PT, νT, CT))
    function dPνC(y, ℙ̃, s) # note interchanged order of arguments
        access = Val{}(dim(ℙ̃))
        P, ν, C = static_accessor_HFc(y, access)
        _B, _β, _σ, _a = Bridge.B(s, ℙ̃), Bridge.β(s, ℙ̃), Bridge.σ(s, ℙ̃), Bridge.a(s, ℙ̃)
    
        dP =  (_B * P) + (P * _B') - _a
        dν =  (_B * ν) + _β
        F = (P \ ν)
        dC = dot(_β, F) + 0.5*Bridge.outer(F' * _σ) - 0.5*tr( (P \ (_a)))
        vectorise(dP, dν, dC)
    end
    yT = vectorise(PT, νT, CT)
    prob = ODEProblem{false}(
            dPνC,   # increment
            yT, # starting val
            (tt[end], tt[1]),   # time interval
            ℙ̃)  # parameter
    access = Val{}(dim(ℙ̃))
    TP = typeof(PT); Tν= typeof(νT); Tc = typeof(CT)
    saved_values = SavedValues(Float64, Tuple{TP,Tν,Tc})
    callback = SavingCallback(
        (u,t,integrator) -> static_accessor_HFc(u, access),
        saved_values;
        saveat=reverse(tt), 
        tdir=-1
    )
    integrator = init(
        prob,
        D.solvertype,
        callback=callback,
        save_everystep=false, # to prevent wasting memory allocations
    )
    sol = DifferentialEquations.solve!(integrator)   # s
    
    savedt = saved_values.t
    ss = saved_values.saveval
    reverse!(ss)
    for i ∈ eachindex(savedt)
        Pt[i] = getindex.(ss,1)[i]
        νt[i] = getindex.(ss,2)[i]
    end
    C = sol.u[1][end]    # = getindex.(saved_y,3)[1]

    Pt, νt, C
end

# to adjust 
function pbridgeode!(D::AssumedDensityFiltering, ℙ, t, (Pt, νt), (PT, νT, CT))
    function dPνC(y, ℙ, s) # note interchanged order of arguments
        access = Val{}(dim(ℙ))
        P, ν, C = static_accessor_HFc(y, access)

        _B = jacobianb(s,ν,ℙ) # should be a function of (s,x)
        #_β = Bridge.b(s,x,ℙ) - _B*x
        _σ = Bridge.σ(s,ν,ℙ)
        _a = Bridge.a(s,ν,ℙ)

    
        dP =  (_B * P) + (P * _B') - _a       # originally - _a
        dν =  Bridge.b(s, ν, ℙ)
        F = (P \ ν)
        dC =  0.5*Bridge.outer(F' * _σ) - 0.5*tr( (P \ (_a))) #+ dot(_β, F) # CHECK
        vectorise(dP, dν, dC)
    end
    yT = vectorise(PT, νT, CT)
    prob = ODEProblem{false}(
            dPνC,   # increment
            yT, # starting val
            (tt[end], tt[1]),   # time interval
            ℙ)  # parameter
    access = Val{}(dim(ℙ))
    TP = typeof(PT); Tν= typeof(νT); Tc = typeof(CT)
    saved_values = SavedValues(Float64, Tuple{TP,Tν,Tc})
    callback = SavingCallback(
        (u,t,integrator) -> static_accessor_HFc(u, access),
        saved_values;
        saveat=reverse(tt), 
        tdir=-1
    )
    # integrator = init(
    #     prob,
    #     D.solvertype,
    #     callback=callback,
    #     save_everystep=false, # to prevent wasting memory allocations
    # )
    # sol = DifferentialEquations.solve!(integrator)   # s
    
    # test
    sol = DifferentialEquations.solve!(init(prob, D.solvertype, callback=callback, save_everystep=false))

    savedt = saved_values.t
    ss = saved_values.saveval
    reverse!(ss)
    for i ∈ eachindex(savedt)
        Pt[i] = getindex.(ss,1)[i]
        νt[i] = getindex.(ss,2)[i]
    end
    C = sol.u[1][end]    # = getindex.(saved_y,3)[1]

    Pt, νt, C
end



function pbridgeode_HFC!(D::DE, ℙ̃, tt, (Ht, Ft), (HT, FT, CT))
    function dHFC(y, ℙ̃, s) # note interchanged order of arguments
        access = Val{}(dim(ℙ̃))
        H, F, C = static_accessor_HFc(y, access)
        _B, _β, _σ, _a = Bridge.B(s, ℙ̃), Bridge.β(s, ℙ̃), Bridge.σ(s, ℙ̃), Bridge.a(s, ℙ̃)
    
        dH = - (_B' * H)  - (H * _B) + Bridge.outer( H * _σ)
        dF = - (_B' * F) + H * (_a * F + _β) 
        dC = dot(_β, F) + 0.5*Bridge.outer(F' * _σ) - 0.5*tr( (H* (_a)))
        vectorise(dH, dF, dC)
    end
    yT = vectorise(HT, FT, CT)
    prob = ODEProblem{false}(
            dHFC,   # increment
            yT, # starting val
            (tt[end], tt[1]),   # time interval
            ℙ̃)  # parameter
    access = Val{}(dim(ℙ̃))
    TP = typeof(HT); Tν= typeof(FT); Tc = typeof(CT)
    saved_values = SavedValues(Float64, Tuple{TP,Tν,Tc})
    callback = SavingCallback(
        (u,t,integrator) -> static_accessor_HFc(u, access),
        saved_values;
        saveat=reverse(tt), 
        tdir=-1
    )
    integrator = init(
        prob,
        D.solvertype,
        callback=callback,
        save_everystep=false, # to prevent wasting memory allocations
    )
    sol = DifferentialEquations.solve!(integrator)   # s
    
    savedt = saved_values.t
    ss = saved_values.saveval
    reverse!(ss)
    for i ∈ eachindex(savedt)
        Ht[i] = getindex.(ss,1)[i]
        Ft[i] = getindex.(ss,2)[i]
    end
    C = sol.u[1][end]    # = getindex.(saved_y,3)[1]

    Ht, Ft, C
end



################################################################################

function pbridgeode!(::Adaptive, ℙ, t, (Pt, νt), (PT, νT, CT), X::SamplePath)
    function dPνC(s, y, (ℙ,x))
        access = Val{}(dim(ℙ))
        P, ν, _ = static_accessor_HFc(y, access)
        #_B, _β, _σ, _a = Bridge.B(s, ℙ̃), Bridge.β(s, ℙ̃), Bridge.σ(s, ℙ̃), Bridge.a(s, ℙ̃)
        _B = jacobianb(s,x,ℙ)
        _β = Bridge.b(s,x,ℙ) - _B*x
        _σ = Bridge.σ(s,x,ℙ)
        _a = Bridge.a(s,x,ℙ)

        dP =  (_B * P) + (P * _B') - _a
        dν =  (_B * ν) + _β
        F = (P \ ν)
        dC = dot(_β, F) + 0.5*Bridge.outer(F' * _σ) - 0.5*tr( (P \ (_a)))
        vectorise(dP, dν, dC)
    end

    Pt[end] = PT
    νt[end] = νT
    C = CT

    access = Val{}(dim(ℙ))
    y = vectorise(PT, νT, CT)

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]

        x = X.yy[i]
    #            s = t[i+1]

        y = kernelrk4(dPνC, t[i+1], y, dt, (ℙ,x))
        Pt[i], νt[i], C = static_accessor_HFc(y, access)
    end

    Pt, νt, C
end



# do this later
    #automatic differentiation jacobian of b at (t,x)
    if false
        ff(s,ℙ) = (u) -> Bridge.b(s,u,ℙ)
        s = 1.0
        x = vT
        ff(s,ℙ)(x)
        #B = jacobian(u -> ff(s,ℙ)(u), x)
        B = jacobian(ff(s,ℙ), x)
    end



function forwardguide!((X, W, ll), (Xᵒ, Wᵒ, Wbuffer), 𝒫, ρ; skip=sk, verbose=false)
    acc = false
    sample!(Wbuffer, wienertype(𝒫.ℙ))
    Wᵒ.yy .= ρ*W.yy + sqrt(1.0-ρ^2)*Wbuffer.yy
    x0 = X.yy[1]
    solve!(Euler(),Xᵒ, x0, Wᵒ, 𝒫)
    llᵒ = llikelihood(Bridge.LeftRule(), Xᵒ, 𝒫, skip=skip)

    if !verbose
        print("ll $ll $llᵒ, diff_ll: ",round(llᵒ-ll;digits=3))
    end
    if log(rand()) <= llᵒ - ll
        X.yy .= Xᵒ.yy
        W.yy .= Wᵒ.yy
        ll = llᵒ
        if !verbose   print("✓")    end
        acc = true
    end
    println()
    (X, W, ll), acc 
end


"""
    forwardguide(x0, ℐ::PathInnovation, 𝒫, ρ; skip=sk, verbose=false)

    returns tuple (ℐ, xend, acc) where
    ℐ:: PathInnovation (updated elements for X, W and ll in case of acceptance, else just the 'input' ℐ)
    xend: endpoint of updated samplepath x
    acc: Booolean if pCN step was accepted
"""
function forwardguide(x0, ℐ::PathInnovation , 𝒫, ρ; skip=sk, verbose=false)
    W, ll, Xᵒ, Wᵒ, Wbuf = ℐ.W, ℐ.ll, ℐ.Xᵒ, ℐ.Wᵒ, ℐ.Wbuf
    sample!(Wbuf, wienertype(𝒫.ℙ))
    Wᵒ.yy .= ρ*W.yy + sqrt(1.0-ρ^2)*Wbuf.yy
    solve!(Euler(),Xᵒ, x0, Wᵒ, 𝒫)
    llᵒ = llikelihood(Bridge.LeftRule(), Xᵒ, 𝒫, skip=skip)

    if !verbose
        print("ll $ll $llᵒ, diff_ll: ",round(llᵒ-ll;digits=3))
    end
    if log(rand()) <= llᵒ - ll
        if !verbose   print("✓")    end
        println()
        return (PathInnovation(Xᵒ, Wᵒ, llᵒ, Xᵒ, Wᵒ, Wbuf), lastval(Xᵒ), true)
    else
        println()
        return (ℐ, lastval(X), false)
    end
end

lastval(X::SamplePath) = X.yy[end]
lastval(ℐ::PathInnovation) = lastval(ℐ.X)

function mergepaths(ℐs)
    tt = map(x->x.X.tt, ℐs)
    yy = map(x->x.X.yy, ℐs)
    SamplePath(vcat(tt...),vcat(yy...))
end