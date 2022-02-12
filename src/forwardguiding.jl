logh̃(x, h0) =  dot(x, -0.5 * h0.H * x + h0.F) - h0.C    

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
                σ̃ = Bridge.σ(tt[i], ℙ̃)
                ll += 0.5*Bridge.inner( σ' * r) * dt    # |σ(t,x)' * tilder(t,x)|^2
                ll -= 0.5*Bridge.inner(σ̃' * r) * dt   # |tildeσ(t)' * tilder(t,x)|^2
                a = Bridge.a(tt[i], x, ℙ)
                ã = Bridge.a(tt[i], ℙ̃)
                ll += 0.5*dot(a-ã, M.H[i]) * dt
            end
        end
        x  +=  b * dt + σ* (σ' * r * dt + dz) 
        push!(X, copy(x))
    end
    X, ll
end


function forwardguide(x0, ℙ, Z::Innovations, B::BackwardFilter; include_h0=true)
    X, ll = forwardguide(x0, ℙ, Z.z[1], B.Ms[1])
    xlast = X[end]
    XX = [X]
    for i in 2:length(B.Ms)
        X, ll_ = forwardguide(xlast, ℙ, Z.z[i], B.Ms[i])
        ll += ll_
        push!(XX, copy(X))
        xlast = X[end]
    end
    ll += logh̃(x0, B.h0) * include_h0
    XX, ll
end

forwardguide(B, ℙ) = (x0, Z) -> forwardguide(x0, ℙ, Z, B);  

