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

    # specialised function for JansenRitDiffusionAux
    function dHFC(y, ℙ̃::JansenRitDiffusionAux, s) # note interchanged order of arguments
        access = Val{}(dim(ℙ̃))
        H, F, _ = static_accessor_HFc(y, access)
        _B, _β = Bridge.B(s, ℙ̃), Bridge.β(s, ℙ̃)
     
        dH = - (_B' * H)  - (H * _B) + Bridge.outer( mulXσ(H, ℙ̃) )
        dF = - (_B' * F) + H * (mulax(F, ℙ̃)  + _β) 
        dC = dot(_β, F) + 0.5* dotσx(F, ℙ̃)^2 - 0.5* trXa(H, ℙ̃)
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


function backwardfiltering(obs, timegrids, ℙ, ℙ̃s ;ϵ = 10e-2)
    n = length(obs)-1
    (HT, FT, CT) = HFC(obs[end])
    Ms = Message[]
    for i in n:-1:1
        M = Message(DE(Vern7()), ℙ, ℙ̃s[i], timegrids[i], HT, FT, CT) 
        pushfirst!(Ms, M)
        (HT, FT, CT) = fusion_HFC(HFC0(M), HFC(obs[i]))
    end
    (HT, FT, CT), Ms
end


function backwardfiltering!(Ms, obs; ϵ = 10e-2) 
    n = length(Ms)
    (HT, FT, CT) = HFC(obs[end])
    for i in n:-1:1
        pbridgeode_HFC!(DE(Vern7()), Ms[i].ℙ̃, Ms[i].tt, (Ms[i].H, Ms[i].F), (HT, FT, CT))
        (HT, FT, CT) = fusion_HFC(HFC0(Ms[i]), HFC(obs[i]))
    end
    (HT, FT, CT)
end

"""
    update_Message(M, tup)

    Construct new instance of Message, with fields in ℙ and ℙ̃ adjusted according to tup
    
    M = Ms[3]
    tup = (C=3333333.1, A=3311.0)
    Mup = update_Message(M,tup)
"""
function update_Message(M::Message,tup)
    # adjust ℙ
    P_ = M.ℙ
    P_ = setproperties(P_, tup)
    @set! M.ℙ = P_
    # adjust ℙ̃
    P̃_ = M.ℙ̃
    P̃_ = setproperties(P̃_, tup)
    @set! M.ℙ̃ = P̃_
    M
end    


"""
    update_Messagees!(Ms, tup)

    Construct new instance of Message, with fields in ℙ and ℙ̃ adjusted according to tup
    Do this for each element of Ms and write into it

    tup = (C=3333333.1, A=3311.0)
    update_Messagees!(Ms,tup)
"""
function update_Messagees!(Ms, tup)
    for i ∈ eachindex(Ms)
        Ms[i] = update_Message(Ms[i], tup)
    end
end




"""
    parupdate!(Msᵒ, θ, pars::ParInfo,  tuningpars)

    Propose new value for θ and write that into all relevant fields (ℙ and ℙ̃) of Msᵒ
"""
function parupdate!(Msᵒ::Vector{Message}, θ, pars::ParInfo,  tuningpars)
    θᵒ = parameterkernel(θ, tuningpars)  
    tup = (; zip(pars.names, θᵒ)...)  # make named tuple 
    update_Messagees!(Msᵒ,tup)  # adjust all ℙ and ℙ̃ fields in Msᵒ according to tup
    θᵒ
end

   
function parinf(obs, timegrids, x0, pars, tuningpars, ρ, ℙ ; 
                        parupdating=true, guidingterm_with_x1=false, 
                        iterations = 300, skip_it = 10, verbose=false, AuxType=JansenRitDiffusionAux)
    # initialisation
    ρs = fill(ρ, length(timegrids))    
    (H0, F0, C0), Ms = init_auxiliary_processes(AuxType, obs, timegrids, ℙ, x0, guidingterm_with_x1);
    Ps = forwardguide(x0, Ms, ρs);
    ll = loglik(x0, (H0,F0,C0), Ps)
    θ = getpar(Ms, pars)

    # containers
    Psᵒ = deepcopy(Ps) 
    Msᵒ = deepcopy(Ms)

    # saving iterates
    subsamples = 0:skip_it:iterations # for saving paths
    XX = [mergepaths(Ps)]
    θs = [θ]
    lls = [ll]

    recomp = maximum(pars.recomputeguidingterm) # if true, then for par updating the guiding term needs to be recomputed

    accinnov = 0; accpar = 0 
    for iter in 1:iterations  
        θ, Ms, Ps, lls_, (H0, F0, C0), accinnov_, accpar_ = update!(θ, Ms, Ps, Msᵒ, Psᵒ, ll, (H0, F0, C0), x0, recomp, tuningpars;
                             verbose=verbose, parupdating=parupdating)
        ll = last(lls_)

        accpar += accpar_
        accinnov += accinnov_
        push!(θs, copy(θ)) 
        push!(lls, lls_...)
        (iter in subsamples) && println(iter)
        (iter in subsamples) && push!(XX, mergepaths(Ps))
  
        adjust_PNCparamters!(Psᵒ, ρ)
    end
    println("acceptance percentage parameter: ", 100*accpar/iterations)
    println("acceptance percentage innovations: ", 100*accinnov/iterations)
    XX, θs, Ps, lls, (accpar=accpar, accinnov=accinnov)
  end
  




  function update!(θ, Ms, Ps, Msᵒ, Psᵒ, ll, (H0, F0, C0), x0, recomp, tuningpars; verbose=false, parupdating=true)
    accinnov_ = 0 ; accpar_ =0
    forwardguide!(PCN(), Psᵒ, Ps, Ms, x0)
    llᵒ  = loglik(x0, (H0,F0,C0), Psᵒ)
    dll = llᵒ - ll
    !verbose && print("Innovations-PCN update. ll $ll $llᵒ, diff_ll: ",round(dll;digits=3)) 
    if log(rand()) < dll   
        Ps, Psᵒ = Psᵒ,  Ps
        ll = llᵒ
        !verbose && print("✓")    
        accinnov_ = 1 
    end 
    lls_ = [ll]

    if parupdating
        θᵒ =  parupdate!(Msᵒ, θ, pars, tuningpars)
        if recomp                # recomp guiding term if at least one parameter requires recomputing the guiding term
            (H0ᵒ, F0ᵒ, C0ᵒ) = backwardfiltering!(Msᵒ, obs) 
        else
            (H0ᵒ, F0ᵒ, C0ᵒ) = (H0, F0, C0)
        end # so whatever Msᵒ was, it got updated by a new value of θ and all other fields are consistent
        forwardguide!(InnovationsFixed(), Psᵒ, Ps, Msᵒ, x0)  # whatever Psᵒ was, using innovations from Ps and Msᵒ we guide forwards
        llᵒ  = loglik(x0, (H0ᵒ,F0ᵒ,C0ᵒ), Psᵒ) # if guiding term need not be recomputed
        dll = llᵒ - ll 
        !verbose && print("Parameter update. ll $ll $llᵒ, diff_ll: ",round(dll;digits=3)) 
        if  log(rand()) < dll && (getpar(Msᵒ, pars)[1]>10.0)  
            θ = θᵒ
            Ms, Msᵒ = Msᵒ, Ms
            Ps, Psᵒ = Psᵒ,  Ps
            ll = llᵒ
            (H0, F0, C0) = (H0ᵒ, F0ᵒ, C0ᵒ) 
            !verbose && print("✓")  
            accpar_ = 1 
        end   
        push!(lls_, ll)
    end
    θ, Ms, Ps, lls_, (H0, F0, C0), accinnov_, accpar_
end


function adjust_PNCparamters!(Psᵒ, ρ; thresh=0.25)
    for i in eachindex(Psᵒ)
        U = rand()
        u = ρ * (U<thresh) + (U>=thresh)
        ui = Psᵒ[i]
        @set! ui.ρ = u
        Psᵒ[i] = ui
    end
end







function checkcorrespondence(P::PathInnovation, M::Message)
    X, W  =  P.X, P.W
    ll0 = P.ll

    x_ = X.yy[1]
    solve!(Euler(),X, x_, W, M)
    ll = llikelihood(Bridge.LeftRule(), X, M, skip=sk)

    println("paths consistent?", X==P.X)
    println("ll consistent?", abs(ll-ll0) <10e-7)
    println(ll-ll0)
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
