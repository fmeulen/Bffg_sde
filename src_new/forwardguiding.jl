logh̃(x, h0) = -0.5 * x' * h0.H * x + h0.F' * x + h0.C    

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
    tup = (; zip(pars.names, θ)...) 
    setproperties(ℙ, tup)
  end  
  
forwardguide(B, ℙ, pars) = (x0, θ, Z) -> forwardguide(x0, setpar(θ, ℙ, pars), Z, B);
  

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

function checkstate(w,B, ℙ, pars)
    _, ll = forwardguide(B, ℙ, pars)(x0, w.θ, w.Z)
    w.ll ==ll, w.ll-ll
  end
  checkstate(B, ℙ, pars) = (w) -> checkstate(w,B, ℙ, pars)



# par updating
function parupdate!(B, ℙ, pars, x0, θ, Z, ll, XX, Prior; verbose=true)
    accpar_ = false
    θᵒ = K(θ)  
    XXᵒ, llᵒ = forwardguide(B, ℙ, pars)(x0, θᵒ, Z);
    !verbose && printinfo(ll, llᵒ, "par") 
    if log(rand()) < llᵒ-ll + (logpdf(Prior, θᵒ) - logpdf(Prior, θ))[1]
      XX, XXᵒ = XXᵒ, XX
      ll = llᵒ
      θ .= θᵒ
      #th, θᵒ = θᵒ, th
      #copyto!(th, θᵒ)
      accpar_ = true
      !verbose && print("✓")  
    end
    XX, ll, θ, accpar_
end

parupdate!(B, ℙ, pars, XX, Prior; verbose=true)  = (x0, th, Z, ll) -> parupdate!(B, ℙ, pars, x0, th, Z, ll, XX, Prior; verbose=verbose)

# innov updating
function pcnupdate!(B, ℙ, pars, x0, θ, Z, ll, XX, Zbuffer, Zᵒ, ρs; verbose=true)
    accinnov_ = false 
    pcn!(Zᵒ, Z, Zbuffer, ρs, ℙ)
    XXᵒ, llᵒ = forwardguide(B, ℙ, pars)(x0, θ, Zᵒ);
    !verbose && printinfo(ll, llᵒ, "pCN") 
    if log(rand()) < llᵒ-ll
        XX, XXᵒ = XXᵒ, XX
        copy!(Z, Zᵒ)
        ll = llᵒ
        accinnov_ = true
        !verbose && print("✓")  
    end
    XX, ll, accinnov_
end

pcnupdate!(B, ℙ, pars, XX, Zbuffer, Zᵒ, ρs; verbose=true) = (x0, θ, Z, ll) ->    pcnupdate!(B, ℙ, pars, x0, θ, Z, ll, XX, Zbuffer, Zᵒ, ρs; verbose=verbose)

#@enter parupdate!(Be, ℙe, pars, XXe, Prior)(x0, θe, Ze, lle);



