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


r((i,t)::IndexedTime, x, 𝒫::PBridge) = (𝒫.P[i] \ (𝒫.ν[i] - x) )
function Bridge._b((i,t)::IndexedTime, x, 𝒫::PBridge)  
    Bridge.b(t, x, 𝒫.ℙ) + Bridge.a(t, x, 𝒫.ℙ) * r((i,t),x,𝒫)   
end

Bridge.σ(t, x, 𝒫::PBridge) = Bridge.σ(t, x, 𝒫.ℙ)
Bridge.a(t, x, 𝒫::PBridge) = Bridge.a(t, x, 𝒫.ℙ)
Bridge.constdiff(𝒫::PBridge) = Bridge.constdiff(𝒫.ℙ) && Bridge.constdiff(𝒫.ℙ̃)

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
