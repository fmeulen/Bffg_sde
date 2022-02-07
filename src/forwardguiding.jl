logh̃(x, h0) = -0.5 * x' * h0.H * x + h0.F' * x - h0.C    

function forwardguide(x0, ℙ, Z, M::Message)
    ℙ̃ = M.ℙ̃
    tt = M.tt
    X = [x0]
    x = x0
    ll::eltype(x0)  = 0.
    for i ∈ 1:length(tt)-1
        dt = tt[i+1]-tt[i]
        b = Bridge.b(tt[i], x, ℙ) 
        r = M.F[i] - M.H[i] * x 
        σ = Bridge.σ(tt[i], x, ℙ)
        dz = Z.yy[i+1] - Z.yy[i]
        
        # likelihood terms
        if i<=length(tt)-1-sk
            db = b  - Bridge.b(tt[i], x, ℙ̃)
            ll += dot(db, r) * dt
            if !constdiff(ℙ) || !constdiff(ℙ̃)
                σ̃ = Bridge.σ(tt[i], x, ℙ̃)
                ll += 0.5*Bridge.inner( σ' * r) * dt    # |σ(t,x)' * tilder(t,x)|^2
                ll -= 0.5*Bridge.inner(σ̃' * r) * dt   # |tildeσ(t,x)' * tilder(t,x)|^2
                a = Bridge.a(tt[i], x, ℙ)
                ã = Bridge.a(tt[i], x, ℙ̃)
                ll += 0.5*dot(a-ã, M.H[i]) * dt
            end
        end
        x  +=  b * dt + σ* (σ' * r * dt + dz) 
        push!(X, copy(x))
    end
    X, ll
end


function forwardguide(x0, ℙ, Z::Innovations, B::BackwardFilter)
    X, ll = forwardguide(x0, ℙ, Z.z[1], B.Ms[1])
    xlast = X[end]
    XX = [X]
    for i in 2:length(B.Ms)
        X, ll_ = forwardguide(xlast, ℙ, Z.z[i], B.Ms[i])
        ll += ll_
        push!(XX, copy(X))
        xlast = X[end]
    end
    ll += logh̃(x0, B.h0) 
    XX, ll
end



function setpar(θ, ℙ, pars) 
    tup = (; zip(pars.names, θ)...) # try copy here
    setproperties(ℙ, tup)
  end  
  
# this one is tricky:
#forwardguide(B, ℙ, pars) = (x0, θ, Z) -> forwardguide(x0, setpar(θ, ℙ, pars), Z, B);
# safer:
forwardguide(B, ℙ) = (x0, Z) -> forwardguide(x0, ℙ, Z, B);  


function parameterkernel(θ, tuningpars, s) 
    shortrange = rand()>s
    Δ = shortrange ?  rand(MvNormal(tuningpars.short)) : rand(MvNormal(tuningpars.long))
    θ + Δ
  end
  
parameterkernel(tuningpars; s=0.33) = (θ) -> parameterkernel(θ, tuningpars, s) 
  



function adjust_PNCparamters!(ρs, ρ; thresh=0.25)
    for i in eachindex(ρs)
        U = rand()
        ρs[i] = ρ * (U<thresh) + (U>=thresh)
    end
end




function pcn!(Zᵒ, Z, Zbuffer, ρs, ℙ)
    noisetype = wienertype(ℙ)
    for i in eachindex(Z.z)
      sample!(Zbuffer.z[i], noisetype)
      Zᵒ.z[i].yy .= ρs[i]*Z.z[i].yy + sqrt(1.0-ρs[i]^2)*Zbuffer.z[i].yy
    end
end
  
function printinfo(ll, llᵒ, s::String) 
    println(s * " update. ll $ll; llᵒ $llᵒ, difference: ",round(llᵒ-ll;digits=3)) 
    #println()
end
  
function copy!(Z1::Innovations, Z2::Innovations)
    for i in eachindex(Z1.z)
        Z1.z[i].yy .= Z2.z[i].yy 
    end
end

function copy!(Z1::Innovations{T}, Z2::Innovations{T}) where {T<:SamplePath}
    for i in eachindex(Z1.z)
        Z1.z[i].yy .= Z2.z[i].yy 
    end
end


import Base.copy
copy(Z::Innovations) = Innovations(deepcopy(Z.z))

function checkstate(w, B, ℙ)
    _, ll = forwardguide(B, ℙ)(x0, w.Z)
    w.ll ==ll, w.ll-ll
end
checkstate(B, ℙ) = (w) -> checkstate(w,B, ℙ)


recomp(pars) = maximum(pars.recomputeguidingterm)


# par updating, θ and XX are overwritten when accepted
function parupdate!(B, ℙ, pars, x0, θ, Z, ll, XX, K, Prior, obs, obsvals, S, AuxType, timegrids; verbose=true)
    accpar_ = false
    θᵒ = K(θ)   
    recompguidingterm = recomp(pars)
    ℙᵒ = setpar(θᵒ, ℙ, pars)    
    if recompguidingterm        
        Bᵒ = BackwardFilter(S, ℙᵒ, AuxType, obs, obsvals, timegrids);
    else 
        Bᵒ = B
    end
    #XXᵒ, llᵒ = forwardguide(Bᵒ, ℙᵒ, pars)(x0, θᵒ, Z);
    XXᵒ, llᵒ = forwardguide(Bᵒ, ℙᵒ)(x0, Z)
    !verbose && printinfo(ll, llᵒ, "par") 
    if log(rand()) < llᵒ-ll + logpdf(Prior, θᵒ) - logpdf(Prior, θ)
      @. XX = XXᵒ
      ll = llᵒ
      @. θ = θᵒ
      B = Bᵒ
      ℙ = ℙᵒ
      accpar_ = true
      !verbose && print("✓")  
    end
    ll, B, ℙ, accpar_
end

parupdate!(B, ℙ, pars, XX, K, Prior, obs, obsvals, S, AuxType, timegrids; verbose=true)  = (x0, θ, Z, ll) -> parupdate!(B, ℙ, pars, x0, θ, Z, ll, XX, K, Prior, obs, obsvals, S, AuxType, timegrids; verbose=verbose)


# innov updating, XX and Z may get overwritten
function pcnupdate!(B, ℙ, pars, x0, θ, Z, ll, XX, Zbuffer, Zᵒ, ρs; verbose=true)
    accinnov_ = false 
    pcn!(Zᵒ, Z, Zbuffer, ρs, ℙ)
    #XXᵒ, llᵒ = forwardguide(B, ℙ, pars)(x0, θ, Zᵒ);
    XXᵒ, llᵒ = forwardguide(B, ℙ)(x0, Zᵒ);
    
    !verbose && printinfo(ll, llᵒ, "pCN") 
    if log(rand()) < llᵒ-ll
        @. XX = XXᵒ
        copy!(Z, Zᵒ)
        ll = llᵒ
        accinnov_ = true
        !verbose && print("✓")  
    end
    ll, accinnov_
end

pcnupdate!(B, ℙ, pars, XX, Zbuffer, Zᵒ, ρs; verbose=true) = (x0, θ, Z, ll) ->    pcnupdate!(B, ℙ, pars, x0, θ, Z, ll, XX, Zbuffer, Zᵒ, ρs; verbose=verbose)

# this is what we want 
function exploremoveσfixed!(B, ℙ, pars, Be, ℙe, parse, x0, θ, Z, ll, XX, Zᵒ, w, Prior; verbose=true) # w::State proposal from exploring chain
    accswap_ = false
    # propose from exploring chain in target chain
    copy!(Zᵒ, w.Z) 
    θᵒ = copy(w.θ)
    ℙᵒ = setpar(θᵒ, ℙ, parse) 
    XXᵒ, llᵒ = forwardguide(B, ℙᵒ)(x0, Zᵒ)
    
    # compute log proposalratio, numerator should be πᵗ(θ, Z)
    ℙeprop = setpar([θ[1]], ℙe, parse)
    _, llproposal = forwardguide(Be, ℙeprop)(x0, Z)
    
    # denominator should be πᵗ(θᵒ, Zᵒ)
    # ℙeᵒ = setpar(θᵒ, ℙe, parse)
    # _, llproposalᵒ = forwardguide(Be, ℙeᵒ)(x0, Zᵒ);
    # llproposalᵒ == w.ll  # should be true
    llproposalᵒ = w.ll

    A = llᵒ - ll +   logpdf(Prior, [θᵒ[1], θ[2]]) - logpdf(Prior, θ)+llproposal - llproposalᵒ
    if log(rand()) < A
        println(θᵒ)
        @show llᵒ - ll 
        @show llproposal - llproposalᵒ 
        @show llproposal
        @show llproposalᵒ
        @show logpdf(Prior, [θᵒ[1], θ[2]]) - logpdf(Prior, θ)
        println()


        @. XX = XXᵒ
        copy!(Z, Zᵒ)
        ll = llᵒ
        ℙ = ℙᵒ
        #@. θ = θᵒ
        θ[1] = θᵒ[1]
        
        accswap_ = true
        !verbose && print("✓")  

    end
    ll, ℙ, accswap_
end

exploremoveσfixed!(B, ℙ, pars, Be, ℙe, parse, XX, Zᵒ, w, Prior; verbose=true) = (x0, θ, Z, ll) ->  exploremoveσfixed!(B, ℙ, pars, Be, ℙe, parse, x0, θ, Z, ll, XX, Zᵒ, w, Prior; verbose=verbose) # w::State proposal from exploring chain

###### now also move σ
# function exploremove!(B, ℙ, pars, Be, ℙe, parse, x0, θ, Z, ll, XX, Zᵒ, w; verbose=true) # w::State proposal from exploring chain
#     accswap_ = false
#     copy!(Zᵒ, w.Z) # proppose from exploring chain in target chain
#     θᵒ = [copy(w.θ)[1], ℙe.σy]
#     ℙᵒ = setpar(θᵒ, ℙ, pars)   # hence in proposal σ=σ_exploring
#     XXᵒ, llᵒ = forwardguide(Be, ℙᵒ, pars)(x0, θᵒ, Zᵒ);  # this is subtle only update C, that why parse and not pars
#     # compute log proposalratio
#     _, llproposal = forwardguide(Be, ℙ, pars)(x0, θ, Z);
#     #_, llproposalᵒ = forwardguide(Be, ℙe, parse)(x0, θᵒ, Zᵒ);
#     llproposalᵒ = w.ll
#     A = llᵒ -ll + llproposal - llproposalᵒ 
#     if log(rand()) < A
#         @. XX = XXᵒ
#         copy!(Z, Zᵒ)
#         ll = llᵒ
#         ℙ = ℙᵒ
#         @. θ = θᵒ
#         B = Be

#         accswap_ = true
#         !verbose && print("✓")  
#     end
#     ll, ℙ, B, accswap_
# end


# exploremove!(B, ℙ, pars, Be, ℙe, parse, XX, Zᵒ, w; verbose=true) = (x0, θ, Z, ll) ->  exploremove!(B, ℙ, pars, Be, ℙe, parse, x0, θ, Z, ll, XX, Zᵒ, w; verbose=verbose) # w::State proposal from exploring chain



# θ = [40.0, 1500.0]
# ℙᵒ = setpar(θ, ℙ, pars)    
# Bᵒ = BackwardFilter(S, ℙᵒ, AuxType, obs, obsvals, timegrids);
# XX, ll = forwardguide(Bᵒ, ℙ, pars)(x0, θ, Z);
# θ, ll

# θᵒ = K(θ)   
# ℙᵒ = setpar(θᵒ, ℙ, pars)    
# Bᵒ = BackwardFilter(S, ℙᵒ, AuxType, obs, obsvals, timegrids);
# XXᵒ, llᵒ = forwardguide(Bᵒ, ℙ, pars)(x0, θᵒ, Z);
# θᵒ, llᵒ, llᵒ - ll

