
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


function parameterkernel(θ, tuningpars, s) 
    shortrange = rand()>s
    Δ = shortrange ?  rand(MvNormal(tuningpars.short)) : rand(MvNormal(tuningpars.long))
    θ + Δ
  end
  
parameterkernel(tuningpars; s=0.33) = (θ) -> parameterkernel(θ, tuningpars, s) 
  

"""
    parupdate!(Msᵒ, θ, pars::ParInfo,  tuningpars)

    Propose new value for θ and write that into all relevant fields (ℙ and ℙ̃) of Msᵒ
"""
function parupdate!(Msᵒ::Vector{Message}, θ, pars::ParInfo,  K)
    θᵒ = K(θ)  
    tup = (; zip(pars.names, θᵒ)...)  # make named tuple 
    update_Messagees!(Msᵒ,tup)  # adjust all ℙ and ℙ̃ fields in Msᵒ according to tup
    θᵒ
end



function inference(obs, timegrids, x0, pars, K, Ke, ρ, ℙ, temperature ; 
    parupdating=true, guidingterm_with_x1=false, 
    iterations = 300, skip_it = 10, verbose=true, AuxType=JansenRitDiffusionAux)

    
    S = ChainState(ρ, timegrids, obs, ℙ, x0, pars, guidingterm_with_x1; AuxType=AuxType);
    
    steep = true

    steep && ( ℙe = setproperties(ℙ, (σy=temperature*ℙ.σy))  )
    steep && (Se = ChainState(ρ, timegrids, obs, ℙe, x0, pars, guidingterm_with_x1; AuxType=AuxType);  )
#@enter update(S, recomp, pars, tuningpars, true, false)
    

    # saving iterates
    subsamples = 0:skip_it:iterations # for saving paths
    XX = [mergepaths(Ps)]
    θs = [getpar(S.Ms, pars)]
    lls = [S.ll]
    steep && (θse = [getpar(Se.Ms, pars)]  )
    steep && (llse = [Se.ll]  )

    recomp = maximum(pars.recomputeguidingterm) # if true, then for par updating the guiding term needs to be recomputed

    accinnov = 0; accpar = 0 
    accinnove = 0; accpare = 0 
    for iter in 1:iterations  
        S, lls_, accinnov_, accpar_ = update(S, recomp, x0, pars, K, parupdating, verbose)
        accpar += accpar_
        accinnov += accinnov_
        push!(θs, copy(S.θ)) 
        push!(lls, lls_...)

        if steep
            Se, lls_e, accinnov_e, accpar_e = update(Se, recomp, x0, pars, Ke, parupdating, verbose)
            accpare += accpar_e
            accinnove += accinnov_e
            push!(θse, copy(Se.θ)) 
            push!(llse, lls_e...)
        end
        (iter in subsamples) && println(iter)
        (iter in subsamples) && push!(XX, mergepaths(S.Ps))
        adjust_PNCparamters!(S.Psᵒ, ρ) # FIXME
    end
    println("acceptance percentage parameter: ", 100*accpar/iterations)
    println("acceptance percentage innovations: ", 100*accinnov/iterations)
    XX, θs, S, lls, (accpar=accpar, accinnov=accinnov), θse, Se, llse
end



function update(S, recomp, x0, pars, K, parupdating, verbose)
    Ms, Ps, Msᵒ, Psᵒ, ll, h0, θ = S.Ms, S.Ps, S.Msᵒ, S.Psᵒ, S.ll, S.h0, S.θ
    
    accinnov_ = 0 ; accpar_ =0
    forwardguide!(PCN(), Psᵒ, Ps, Ms, x0)
    llᵒ  = loglik(x0, h0, Psᵒ)
    dll = llᵒ - ll
    !verbose && print("Innovations-PCN update. ll $ll $llᵒ, diff_ll: ",round(dll;digits=3)) 
    if log(rand()) < dll   
#        Ps, Psᵒ = Psᵒ,  Ps
        for i in eachindex(Ps)
            Ps[i] = PathInnovation(Psᵒ[i])
        end
        ll = llᵒ
        !verbose && print("✓")    
        accinnov_ = 1 
    end 
    lls_ = [ll]

    if parupdating
        θᵒ =  parupdate!(Msᵒ, θ, pars, K)
        if recomp                # recomp guiding term if at least one parameter requires recomputing the guiding term
            h0ᵒ = backwardfiltering!(Msᵒ, obs) 
        else
            h0ᵒ = h0
        end # so whatever Msᵒ was, it got updated by a new value of θ and all other fields are consistent
        forwardguide!(InnovationsFixed(), Psᵒ, Ps, Msᵒ, x0)  # whatever Psᵒ was, using innovations from Ps and Msᵒ we guide forwards
        llᵒ  = loglik(x0, h0ᵒ, Psᵒ) # if guiding term need not be recomputed
        dll = llᵒ - ll 
        !verbose && print("Parameter update. ll $ll $llᵒ, diff_ll: ",round(dll;digits=3)) 
        if  log(rand()) < dll && (getpar(Msᵒ, pars)[1]>10.0)  
            θ = θᵒ
            Ms, Msᵒ = Msᵒ, Ms
            #Ps, Psᵒ = Psᵒ,  Ps
            for i in eachindex(Ps)
                Ps[i] = PathInnovation(Psᵒ[i])
            end
     
            ll = llᵒ
            h0 = h0ᵒ
            !verbose && print("✓")  
            accpar_ = 1 
        end   
        push!(lls_, ll)
    end
    ChainState(Ms, Ps, Msᵒ, Psᵒ, ll, h0, θ), lls_, accinnov_, accpar_
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






