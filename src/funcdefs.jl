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

    function PathInnovation(x0, 𝒫, ρ)
        tt = 𝒫.tt
        W = sample(tt, wienertype(𝒫.ℙ))    
        X = solve(Euler(), x0, W, 𝒫)  # allocation        
        ll = llikelihood(Bridge.LeftRule(), X, 𝒫, skip=sk)
        Wbuf = deepcopy(W)
        #PathInnovation(X, W, ll, Wbuf, ρ)
        new{typeof(X), typeof(W), typeof(ll)}(X, W, ll, Wbuf, ρ)
    end
end


"""
    GuidedProcess

        struct for guide process on a segment (equivalently kernel)
    ℙ:  target diffusion
    ℙ̃:  auxiliary NclarDiffusion
    tt: time grid for diffusion (including start and end time)
    H:  H-values on tt
    F:  F values on tt
    C:  -C is an additive factor in the loglikelihood

    constructors for solving the backward filtering by numerically approximating ODEs
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

    # constructor: provide (ℙ, ℙ̃, timegrid HT, FT, CT)    
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
        #GuidedProcess(ℙ, ℙ̃, tt, Ht, Ft, C)
        new{eltype(Ft), typeof(ℙ), typeof(ℙ̃), eltype(Ht), eltype(Ft), typeof(C)}(ℙ, ℙ̃, tt, Ht, Ft, C)
    end
end

function convert_PνC_to_HFC(P,ν,C)
    H = inv(P)
    H, P\ν, C
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
    #    saveat=tt, 
        tdir=-1
    )
    integrator = init(
        prob,
        D.solvertype,
        callback=callback,
        save_everystep=false, # to prevent wasting memory allocations
    )
    sol = DifferentialEquations.solve!(integrator)   # s
    
    #  savedt = saved_values.t
    ss = saved_values.saveval

    reverse!(ss)
    Ht .= getindex.(ss,1)
    Ft .= getindex.(ss,2)
    # for i ∈ eachindex(savedt)
    #     Ht[i] = getindex.(ss,1)[i]  # trouble with profileview
    #     Ft[i] = getindex.(ss,2)[i]
    # end
    #    C = sol.u[1][end]    # = getindex.(saved_y,3)[1]
    C = getindex(ss[end],3)
    Ht, Ft, C
end




"""
    fusion_HFC((H1, F1, C1), (H2, F2, C2))

    returns added characteristics that correspond to fusion in (H,F,C)-parametrisation
"""
function fusion_HFC((H1, F1, C1), (H2, F2, C2))
    H1 + H2, F1 + F2, C1+C2
end



r((i,t)::IndexedTime, x, 𝒫::GuidedProcess) = 𝒫.F[i] - 𝒫.H[i] * x 

logh̃(x, (H,F,C)) = -0.5 * x' * H * x + F' * x + C    
   
Bridge._b((i,t)::IndexedTime, x, 𝒫::GuidedProcess)  =  Bridge.b(t, x, 𝒫.ℙ) + Bridge.a(t, x, 𝒫.ℙ) * r((i,t),x,𝒫)   
Bridge.σ(t, x, 𝒫::GuidedProcess) = Bridge.σ(t, x, 𝒫.ℙ)
Bridge.a(t, x, 𝒫::GuidedProcess) = Bridge.a(t, x, 𝒫.ℙ)
Bridge.constdiff(𝒫::GuidedProcess) = Bridge.constdiff(𝒫.ℙ) && Bridge.constdiff(𝒫.ℙ̃)


function llikelihood(::LeftRule, X::SamplePath, 𝒫::GuidedProcess; skip = sk)
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
    som 
end

function loglik(x0, (H0,F0,C0), ℐs::Vector{PathInnovation})
    logh̃(x0, (H0,F0,C0)) + sum(map(x -> x.ll, ℐs))
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



function forwardguide(x0, 𝒫s, ρs)
    xend = x0
    ℐs = PathInnovation[]
    for i ∈ eachindex(𝒫s)
        push!(ℐs, PathInnovation(xend, 𝒫s[i], ρs[i]))
        xend = lastval(ℐs[i])
    end
    # H0, F0, C0 = 𝒫s[1].H[1], 𝒫s[1].F[1], 𝒫s[1].C
    # loglik = logh̃(x0, (H0,F0,C0)) + sum(map(x -> x.ll, ℐs))
    ℐs
end


"""
    forwardguide(::InnovationsFixed, ℐ::PathInnovation, 𝒫::GuidedProcess, x0; skip=sk, verbose=false)

    Using GuidedProposal 𝒫 and innovations extracted from the W-field of ℐ, simulate a guided process starting in x0
"""
function forwardguide!(::InnovationsFixed, ℐᵒ::PathInnovation,  ℐ::PathInnovation, 𝒫::GuidedProcess, x0)    
    ℐᵒ.W.yy .= ℐ.W.yy
    solve!(Euler(), ℐᵒ.X, x0, ℐ.W, 𝒫)
    llᵒ = llikelihood(Bridge.LeftRule(), ℐᵒ.X, 𝒫, skip=sk)
    lastval(ℐᵒ), llᵒ
end

function forwardguide!(::PCN, ℐᵒ::PathInnovation,  ℐ::PathInnovation, 𝒫::GuidedProcess, x0)    
    sample!(ℐᵒ.Wbuf, wienertype(𝒫.ℙ))
    ρ = ℐᵒ.ρ
    ℐᵒ.W.yy .= ρ * ℐ.W.yy + sqrt(1.0-ρ^2)*ℐᵒ.Wbuf.yy
    solve!(Euler(), ℐᵒ.X, x0, ℐᵒ.W, 𝒫)
    llᵒ = llikelihood(Bridge.LeftRule(), ℐᵒ.X, 𝒫, skip=sk)
    lastval(ℐᵒ), llᵒ
end

"""
    forwardguide!(gt::GuidType, ℐs::Vector{PathInnovation}, 𝒫s::Vector{GuidedProcess}, x0; skip=sk, verbose=false)

    Using a vector of guided process, simulate a new path on all segments. 
    The elements of ℐs get overwritten and hence possibly change. 

    returns total number of segments on which the update type was accepted.
"""
function forwardguide!(gt::GuidType, ℐsᵒ::Vector{PathInnovation}, ℐs::Vector{PathInnovation}, 𝒫s::Vector{GuidedProcess}, x0)
    x_ = x0  
    xend = 0.0*x0 ; 
    llᴼ = 0.0 
    for i ∈ eachindex(ℐs)
        xend, llᵒ = forwardguide!(gt, ℐsᵒ[i], ℐs[i], 𝒫s[i], x_) # profileview colours red on this line, especially when PCN is called
        x_ = xend
        ui = ℐsᵒ[i]
        @set! ui.ll = llᵒ
        ℐsᵒ[i] = ui
   end
end











function backwardfiltering(obs, timegrids, ℙ, ℙ̃s ;ϵ = 10e-2)
    #Hinit, Finit, Cinit =  init_HFC(obs[end].v, obs[end].L, dim(ℙ); ϵ=ϵ)
    n = length(obs)
    #HT, FT, CT = fusion_HFC(HFC(obs[n]), (Hinit, Finit, Cinit) )
    (HT, FT, CT) = HFC(obs[n])
    𝒫s = GuidedProcess[]

    for i in n-1:-1:1
        𝒫 = GuidedProcess(DE(Vern7()), ℙ, ℙ̃s[i], timegrids[i], HT, FT, CT) # profileview colours red here
        pushfirst!(𝒫s, 𝒫)
        # message = (𝒫.H[1], 𝒫.F[1], 𝒫.C[1])
        # (HT, FT, CT) = fusion_HFC(message, HFC(obs[i]))
        (HT, FT, CT) = fusion_HFC(HFC0(𝒫), HFC(obs[i]))
    end
    (HT, FT, CT), 𝒫s
end

HFC0(𝒫::GuidedProcess) = (𝒫.H[1], 𝒫.F[1], 𝒫.C[1])

function backwardfiltering!(𝒫s, obs, timegrids; ϵ = 10e-2) #FIXME
    #Hinit, Finit, Cinit =  init_HFC(obs[end].v, obs[end].L, dim(𝒫s[1].ℙ); ϵ=ϵ)
    n = length(obs)
    #HT, FT, CT = fusion_HFC(HFC(obs[n]), (Hinit, Finit, Cinit) )
    (HT, FT, CT) = HFC(obs[n])

    for i in n-1:-1:1
        𝒫s[i] = GuidedProcess(DE(Vern7()), 𝒫s[i].ℙ, 𝒫s[i].ℙ̃, timegrids[i], HT, FT, CT)
        # message = (𝒫s[i].H[1], 𝒫s[i].F[1], 𝒫s[i].C[1])
        # (HT, FT, CT) = fusion_HFC(message, HFC(obs[i]))
        (HT, FT, CT) = fusion_HFC(HFC0(𝒫s[i]), HFC(obs[i]))
    end
    #(HT, FT, CT), 𝒫s
    (HT, FT, CT)
end




function parupdate!(𝒫sᵒ, θ, pars::ParInfo,  tuningpars)
    θᵒ = parameterkernel(θ, tuningpars)  
    tup = (; zip(pars.names, θᵒ)...)  # make named tuple 
    update_guidedprocesses!(𝒫sᵒ,tup)  # adjust all ℙ and ℙ̃ fields in 𝒫sᵒ according to tup
    θᵒ
end

 
    




function parinf(obs, timegrids, x0, pars, tuningpars, ρ, ℙ, ℙ̃s; 
                        parupdating=true, iterations = 300, skip_it = 10, verbose=false)
  
    (H0, F0, C0), 𝒫s = backwardfiltering(obs, timegrids, ℙ, ℙ̃s; ϵ = 10e-5);
    ρs = fill(ρ, length(timegrids))    
    ℐs = forwardguide(x0, 𝒫s, ρs);
    ll = loglik(x0, (H0,F0,C0), ℐs)

    # containers
    ℐsᵒ = deepcopy(ℐs) 
    𝒫sᵒ = deepcopy(𝒫s)

    # don't save all paths
    subsamples = 0:skip_it:iterations
    XX = [mergepaths(ℐs)]
 
    θ = getpar(𝒫s, pars)
    θs = [θ]
    
    recomp = maximum(pars.recomputeguidingterm) # if true, then for par updating the guiding term needs to be recomputed

    accinnov = 0; accpar = 0 
    for iter in 1:iterations  
        forwardguide!(PCN(), ℐsᵒ, ℐs, 𝒫s, x0)
        llᵒ  = loglik(x0, (H0,F0,C0), ℐsᵒ)
        dll = llᵒ - ll
        !verbose && print("Innovations-PCN update. ll $ll $llᵒ, diff_ll: ",round(dll;digits=3)) 
        if log(rand()) < dll   
            ℐs, ℐsᵒ = ℐsᵒ,  ℐs
            ll = llᵒ
            !verbose && print("✓")    
            accinnov += 1 
        end 
    
        if parupdating
            θᵒ =  parupdate!(𝒫sᵒ, θ, pars, tuningpars)
            if recomp                # recomp guiding term if at least one parameter requires recomputing the guiding term
                (H0ᵒ, F0ᵒ, C0ᵒ) = backwardfiltering!(𝒫sᵒ, obs, timegrids) 
            else
                (H0ᵒ, F0ᵒ, C0ᵒ) = (H0, F0, C0)
            end
            forwardguide!(InnovationsFixed(), ℐsᵒ, ℐs, 𝒫sᵒ, x0)
            llᵒ  = loglik(x0, (H0ᵒ,F0ᵒ,C0ᵒ), ℐsᵒ) # if guiding term need not be recomputed
            dll = llᵒ - ll 
            !verbose && print("Parameter update. ll $ll $llᵒ, diff_ll: ",round(dll;digits=3)) 
            if  log(rand()) < dll #&& (getpar(𝒫sᵒ[1].ℙ)[1]>60.0)  
                θ = θᵒ
                𝒫s, 𝒫sᵒ = 𝒫sᵒ, 𝒫s
                ℐs, ℐsᵒ = ℐsᵒ,  ℐs
                ll = llᵒ
                (H0, F0, C0) = (H0ᵒ, F0ᵒ, C0ᵒ) 
                !verbose && print("✓")  
                accpar += 1 
            end   
            push!(θs, copy(θ)) 
        end
 
        (iter in subsamples) && println(iter)
        (iter in subsamples) && push!(XX, mergepaths(ℐs))
  
        # adjust PCN updating pars (some segments randomly left unchanged)
        for i in eachindex(ℐsᵒ)
            U = rand()
            u = ρ * (U<0.25) + (U>=0.25)
            ui = ℐsᵒ[i]
            @set! ui.ρ = u
            ℐsᵒ[i] = ui
        end
    end
    println("acceptance percentage parameter: ", 100*accpar/iterations)
    println("acceptance percentage innovations: ", 100*accinnov/iterations)
    XX, θs, ℐs, (accpar=accpar, accinnov=accinnov)
  end
  











function checkcorrespondence(ℐ::PathInnovation, 𝒫::GuidedProcess)
    X, W  =  ℐ.X, ℐ.W
    ll0 = ℐ.ll

    x_ = X.yy[1]
    solve!(Euler(),X, x_, W, 𝒫)
    ll = llikelihood(Bridge.LeftRule(), X, 𝒫, skip=sk)

    println("paths consistent?", X==ℐ.X)
    println("ll consistent?", abs(ll-ll0) <10e-7)
    println(ll-ll0)
end



