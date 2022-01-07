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
        PathInnovation(X,W,ll,Xᵒ, Wᵒ, Wbuf)
        # TX, TW, Tll = typeof(X), typeof(W), typeof(ll)
        # new{TX, TW, Tll}(X,W,ll,Xᵒ, Wᵒ, Wbuf)
    end
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
struct GuidedProcess{T,Tℙ,Tℙ̃,TH,TF,TC} <: ContinuousTimeProcess{T}
    ℙ::Tℙ   
    ℙ̃::Tℙ̃   
    tt::Vector{Float64}  
    H::Vector{TH}      
    F::Vector{TF}      
    C::TC              
    GuidedProcess(ℙ::Tℙ, ℙ̃::Tℙ̃, tt, Ht::Vector{TH}, Ft::Vector{TF}, C::TC) where {Tℙ,Tℙ̃,TH,TF,TC} =
        new{Bridge.valtype(ℙ),Tℙ,Tℙ̃,TH,TF,TC}(ℙ, ℙ̃, tt, Ht, Ft, C)

    # constructor: provide (timegrid, ℙ, ℙ̃, νT, PT, CT)    
    function GuidedProcess(::RK4, ℙ, ℙ̃, tt_, HT::TH, FT::TF, CT) where {TH, TF}
        tt = collect(tt_)
        N = length(tt)
        Ht = zeros(TH, N)
        Ft = zeros(TF, N)
        _, _, C = pbridgeode_HFC!(RK4(), ℙ̃, tt, (Ht, Ft), (HT, FT, CT))
        GuidedProcess(ℙ, ℙ̃, tt, Ht, Ft, C)
    end

    
    function GuidedProcess(D::DE, ℙ, ℙ̃, tt_, HT::TH, FT::TF, CT) where {TH, TF}
        tt = collect(tt_)
        N = length(tt)
        Ht = zeros(TH, N)
        Ft = zeros(TF, N)
        _, _, C = pbridgeode_HFC!(D, ℙ̃, tt, (Ht, Ft), (HT, FT, CT))
        GuidedProcess(ℙ, ℙ̃, tt, Ht, Ft, C)
    end
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



r((i,t)::IndexedTime, x, 𝒫::GuidedProcess) = 𝒫.F[i] - 𝒫.H[i] * x 
logh̃(x, 𝒫::GuidedProcess) = -0.5 * x' * 𝒫.H[1] * x + 𝒫.F[1]' * x - 𝒫.C    
   
Bridge._b((i,t)::IndexedTime, x, 𝒫::GuidedProcess)  =  Bridge.b(t, x, 𝒫.ℙ) + Bridge.a(t, x, 𝒫.ℙ) * r((i,t),x,𝒫)   
Bridge.σ(t, x, 𝒫::GuidedProcess) = Bridge.σ(t, x, 𝒫.ℙ)
Bridge.a(t, x, 𝒫::GuidedProcess) = Bridge.a(t, x, 𝒫.ℙ)
Bridge.constdiff(𝒫::GuidedProcess) = Bridge.constdiff(𝒫.ℙ) && Bridge.constdiff(𝒫.ℙ̃)


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
    X, W, ll, Xᵒ, Wᵒ, Wbuf = ℐ.X, ℐ.W, ℐ.ll, ℐ.Xᵒ, ℐ.Wᵒ, ℐ.Wbuf
    sample!(Wbuf, wienertype(𝒫.ℙ))
    Wᵒ.yy .= ρ*W.yy + sqrt(1.0-ρ^2)*Wbuf.yy
    solve!(Euler(),Xᵒ, x0, Wᵒ, 𝒫)
    llᵒ = llikelihood(Bridge.LeftRule(), Xᵒ, 𝒫, skip=skip)

    !verbose && print("ll $ll $llᵒ, diff_ll: ",round(llᵒ-ll;digits=3))
    if log(rand()) <= llᵒ - ll
        !verbose &&  print("✓")    
        !verbose && println()
        return (PathInnovation(Xᵒ, Wᵒ, llᵒ, Xᵒ, Wᵒ, Wbuf), lastval(Xᵒ), true)
    else
        !verbose && println()
        return (ℐ, lastval(X), false)
    end
end


function forwardguide(x0, ℐs::Vector{PathInnovation} , 𝒫s, ρ; skip=sk, verbose=false)
    acc = 0
    xend = x0  
    for i ∈ 1:n-1
        (ℐs[i], xend, a) = forwardguide(xend, ℐs[i], 𝒫s[i], ρ; skip=skip, verbose=verbose);
        acc += a
    end
    ℐs, acc
end





function backwardfiltering(obs, timegrids, ℙ, ℙ̃ ;ϵ = 10e-2, M)
    Hinit, Finit, Cinit =  init_HFC(obs[end].v, obs[end].L, dim(ℙ); ϵ=ϵ)
    n = length(obs)

    HT, FT, CT = fusion_HFC(HFC(obs[n]), (Hinit, Finit, Cinit) )
    𝒫s = GuidedProcess[]
    for i in n:-1:2
        tt = timegrid(obs[i-1].t, obs[i].t, M=M)
        𝒫 = GuidedProcess(DE(Vern7()), ℙ, ℙ̃, timegrids[i-1], HT, FT, CT)
        pushfirst!(𝒫s, 𝒫)
        message = (𝒫.H[1], 𝒫.F[1], 𝒫.C[1])
        (HT, FT, CT) = fusion_HFC(message, HFC(obs[i-1]))
    end
    (HT, FT, CT), 𝒫s
end




timegrids = set_timegrids(obs,100)
