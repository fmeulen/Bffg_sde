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


function pbridgeode_HFC!(D::DE, ℙ̃, tt, (Ht, Ft), hT)
    function dHFC(y, ℙ̃, s) # note interchanged order of arguments
        access = Val{}(dim(ℙ̃))
        
        H, F, C = static_accessor_HFc(y, access)
        _B, _β, _σ, _a = Bridge.B(s, ℙ̃), Bridge.β(s, ℙ̃), Bridge.σ(s, ℙ̃), Bridge.a(s, ℙ̃)
    
        dH = - (_B' * H)  - (H * _B) + Bridge.outer( H * _σ)
        dF = - (_B' * F) + H * (_a * F + _β) 
        dC = dot(_β, F) + 0.5*Bridge.outer(F' * _σ) - 0.5*tr( (H* (_a)))
        vectorise(dH, dF, dC)
    end

    # specialised function for JansenRitDiffusionAux
    # function dHFC(y, ℙ̃::JansenRitDiffusionAux, s) # note interchanged order of arguments
    #     access = Val{}(dim(ℙ̃))
    #     H, F, _ = static_accessor_HFc(y, access)
    #     _B, _β = Bridge.B(s, ℙ̃), Bridge.β(s, ℙ̃)
     
    #     dH = - (_B' * H)  - (H * _B) + Bridge.outer( mulXσ(H, ℙ̃) )
    #     dF = - (_B' * F) + H * (mulax(F, ℙ̃)  + _β) 
    #     dC = dot(_β, F) + 0.5* dotσx(F, ℙ̃)^2 - 0.5* trXa(H, ℙ̃)
    #     vectorise(dH, dF, dC)
    # end

    
    
    yT = vectorise(hT.H, hT.F, hT.C)
    prob = ODEProblem{false}(
            dHFC,   # increment
            yT, # starting val
            (tt[end], tt[1]),   # time interval
            ℙ̃)  # parameter
    access = Val{}(dim(ℙ̃))
    
    TP = typeof(hT.H); Tν= typeof(hT.F); Tc = typeof(hT.C)
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
    C = getindex(ss[end],3)
    Ht, Ft, C
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


function pbridgeode_HFC!(::RK4, ℙ̃, t, (Ht, Ft), hT)
    function dHFC(s, y, ℙ̃)
        access = Val{}(dim(ℙ̃))
        
        H, F, _ = static_accessor_HFc(y, access)
        _B, _β, _σ, _a = Bridge.B(s, ℙ̃), Bridge.β(s, ℙ̃), Bridge.σ(s, ℙ̃), Bridge.a(s, ℙ̃)

        dH = - (_B' * H)  - (H * _B) + Bridge.outer( H * _σ)
        dF = - (_B' * F) + H * (_a * F + _β) 
        dC = dot(_β, F) + 0.5*Bridge.outer(F' * _σ) - 0.5*tr( (H* (_a)))
        vectorise(dH, dF, dC)
    end

    Ht[end] = hT.H
    Ft[end] = hT.F
    C = hT.C
    access = Val{}(dim(ℙ̃))
    
    y = vectorise(hT.H, hT.F, hT.C)

    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        y = kernelrk4(dHFC, t[i+1], y, dt, ℙ̃)
        Ht[i], Ft[i], C = static_accessor_HFc(y, access)
    end
    Ht, Ft, C
end


function pbridgeode_HFC!(S::Vern7direct, ℙ̃, t, (Ht, Ft), hT)
    function dHFC(s, y, ℙ̃)
        #access = Val{}(dim(ℙ̃))
        access = Val{6}()
        H, F, _ = static_accessor_HFc(y, access)
        _B, _β, _σ, _a = Bridge.B(s, ℙ̃), Bridge.β(s, ℙ̃), Bridge.σ(s, ℙ̃), Bridge.a(s, ℙ̃)

        dH = - (_B' * H)  - (H * _B) + Bridge.outer( H * _σ)
        dF = - (_B' * F) + H * (_a * F + _β) 
        dC = dot(_β, F) + 0.5*Bridge.outer(F' * _σ) - 0.5*tr( (H* (_a)))
        vectorise(dH, dF, dC)
    end

    access = Val{6}()   #access = Val{}(dim(ℙ̃))
    y = vectorise(hT.H, hT.F, hT.C)
    Ht[end], Ft[end], C =  static_accessor_HFc(y, access)
    
    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        y = vern7(dHFC, t[i+1], y, dt, ℙ̃, S.tableau)
        
        Ht[i], Ft[i], C = static_accessor_HFc(y, access)

        @show C
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
    n = length(obs)-1
    hT = obs[end].h
    Ms = [Message(S, ℙ̃s[n], timegrids[n], hT) ]
    for i in (n-1):-1:1
        M = Message(S, ℙ̃s[i], timegrids[i], hT) 
        pushfirst!(Ms, M)
        hT = fusion_HFC(Htransform(M), obs[i].h)
    end
    hT, Ms
end


function backwardfiltering!(S, Ms, obs) 
    n = length(Ms)
    hT = obs[end].h
    for i in n:-1:1
        pbridgeode_HFC!(S, Ms[i].ℙ̃, Ms[i].tt, (Ms[i].H, Ms[i].F), hT) #FIXME  S=DE(Vern7())
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


# tt = [Ms[i].tt for i in eachindex(Ps)]
# yy = [Ms[i].ℙ̃.x1(tt[i]) for i in eachindex(Ps)]
# p = plot_(Ps,1)
# plot!(p,vcat(tt...), vcat(yy...),color="grey")


function init_auxiliary_processes(S, ℙ, AuxType, obs, timegrids, x0, guidingterm_with_x1::Bool; x1_init=0.0)
    i=2
    lininterp = LinearInterpolation([obs[i-1].t, obs[i].t], [x1_init, x1_init] )
    ℙ̃s = [AuxType(obs[i].t, obs[i].v[1], lininterp, false, ℙ)]
    n = length(obs)
    for i in 3:n # skip x0
      lininterp = LinearInterpolation([obs[i-1].t, obs[i].t], [x1_init, x1_init] )
      push!(ℙ̃s, AuxType(obs[i].t, obs[i].v[1], lininterp, false, ℙ))
    end
    h0, Ms = backwardfiltering(S, obs, timegrids, ℙ̃s)
    if guidingterm_with_x1
        add_deterministicsolution_x1!(Ms, x0)
        h0 = backwardfiltering!(S, Ms, obs)
    end
    h0, Ms
end

