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

dHFC_DE(y, ℙ̃, s) = dHFC(s, y, ℙ̃)

function ode_HFC!(D::DE, ℙ̃, tt, (Ht, Ft), hT)
    access = Val{}(dim(ℙ̃))
    TP = typeof(hT.H); Tν= typeof(hT.F); Tc = typeof(hT.C)
    saved_values = SavedValues(Float64, Tuple{TP,Tν,Tc})

    yT = vectorise(hT.H, hT.F, hT.C)
    prob = ODEProblem{false}(
            dHFC_DE,                    # increment
            yT,                         # starting val
            (tt[end], tt[1]),           # time interval
            ℙ̃)                          # parameter
    
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
#    sol = DifferentialEquations.solve!(integrator)   # s
 
    DifferentialEquations.solve!(integrator)   # s
    ss = saved_values.saveval  # these are in reversed order, so C is obtained from the last index and for Ht and Ft we need to reverse
    for i in eachindex(ss)
        Ht[end-i+1] = ss[i][1]
        Ft[end-i+1] = ss[i][2]
    end
    C = ss[end][3]  
    Ht, Ft, C
end

function dHFC(s, y, ℙ̃)
    access = Val{}(dim(ℙ̃))  #
    #access = Val{6}()
    H, F, _ = static_accessor_HFc(y, access)
    _B, _β, _σ, _a = Bridge.B(s, ℙ̃), Bridge.β(s, ℙ̃), Bridge.σ(s, ℙ̃), Bridge.a(s, ℙ̃)

    dH = (- (_B' * H)  - (H * _B) + Bridge.outer( H * _σ))
    dF  = (- (_B' * F) + H * (_a * F + _β) )
    dC =  (dot(_β, F) + 0.5*dot(F, _a, F) - 0.5*tr( (H* (_a))))
    #U = (F' * _σ) ;  dC =  (dot(_β, F) + 0.5*dot(U,U) - 0.5*tr( (H* (_a))))
    vectorise(dH, dF, dC)
end



function ode_HFC!(::RK4, ℙ̃, t, (Ht, Ft), hT)
    access = Val{}(dim(ℙ̃))  ##access = Val{6}()   
    y = vectorise(hT.H, hT.F, hT.C)
    Ht[end], Ft[end], C =  static_accessor_HFc(y, access)
    
    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        y = kernelrk4(dHFC, t[i+1], y, dt, ℙ̃)
        Ht[i], Ft[i], C = static_accessor_HFc(y, access)
    end
    Ht, Ft, C
end

function ode_HFC!(S::Vern7direct, ℙ̃, t, (Ht, Ft), hT)
    access = Val{}(dim(ℙ̃))  ##access = Val{6}()   
    y = vectorise(hT.H, hT.F, hT.C)
    Ht[end], Ft[end], C =  static_accessor_HFc(y, access)
    
    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        y = vern7(dHFC, t[i+1], y, dt, ℙ̃, S.tableau)
        Ht[i], Ft[i], C = static_accessor_HFc(y, access)
    end
    Ht, Ft, C
end






"""
    fusion_HFC(h1, h2)

    returns added characteristics that correspond to fusion in (H,F,C)-parametrisation
"""
fusion_HFC(h1, h2) = Htransform(h1.H + h2.H, h1.F + h2.F, h1.C + h2.C)


############# here rewrite with handwritten vern function
function backwardfiltering(S,obs, timegrids, ℙ̃s)
    hT = obs[end].h 
    M = Message(S, ℙ̃s[end], timegrids[end], hT) 
    Ms = [M]
    nseg = length(timegrids)
    for i in nseg:-1:2
        hT = fusion_HFC(Htransform(M), obs[i].h)
        M = Message(S, ℙ̃s[i-1], timegrids[i-1], hT) 
        pushfirst!(Ms, M)   
    end
    hT = Htransform(M)
#    hT = fusion_HFC(Htransform(M), obs[1].h) # only if x0 is not fully observed
    hT, Ms
end



function backwardfiltering!(S, Ms, obs) ##FIXME
    n = length(Ms)
    hT = obs[end].h
    for i in n:-1:1
        ode_HFC!(S, Ms[i].ℙ̃, Ms[i].tt, (Ms[i].H, Ms[i].F), hT) #FIXME  S=DE(Vern7())
        hT = fusion_HFC(Htransform(Ms[i]), obs[i].h)
    end
    hT
end


# solving for deterministic system in (x1, x4)

"""
    odesolx1(t, (x10, x40),  ℙ::JansenRitDiffusionAux)

    We consider the first and fourth coordinate of the JR-system, equating the difference 
    x2-x3 to the observed value at the right-end-point of the time-interval.

    On the timegrid t, the solution for x1 is computed, provided the initial conditions (x10, x40) at time t[1]

    Returns:
    - solution of x1 on timegrid t
    - (x1, x4)) at t[end]
"""

function odesolx1(t, (x10, x40),  ℙ::JansenRitDiffusionAux)
    t0 = t[1]
    vT = ℙ.vT[1]
    c = ℙ.A*ℙ.a*sigm(vT, ℙ)
    k1= x10 - c/ℙ.a^2
    k2 = x40 + ℙ.a*k1 
    dt = t .- t0
    sol = c/ℙ.a^2 .+ (k1 .+ k2* dt) .* exp.(-ℙ.a*dt) 
    x4end = (k2- ℙ.a *k1-ℙ.a*k2*(t[end]-t0)) * exp(-ℙ.a*(t[end]-t0))
    sol, (sol[end], x4end)
end

"""
    add_deterministicsolution_x1!(Ms::Vector{Message}, x0)

    Sequentially call (on each segment)
        odesolx1(t, (x10, x40),  ℙ::JansenRitDiffusionAux)
    such that the resulting path is continuous. 

    Write the obtained solution for x1 into the ℙ̃.x1 field on each Message
"""
function add_deterministicsolution_x1!(Ms::Vector{Message}, x0)
    xend = x0
    for i in eachindex(Ms)
        u = Ms[i]
        sol, xend = odesolx1(u.tt, xend, u.ℙ̃)
        @set! u.ℙ̃.x1 = LinearInterpolation(u.tt, sol)
        @set! u.ℙ̃.guidingterm_with_x1 = true
        Ms[i] = u 
    end
end

