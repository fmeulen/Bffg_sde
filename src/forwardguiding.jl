

r((i,t)::IndexedTime, x, M::Message) = M.F[i] - M.H[i] * x 
Bridge._b((i,t)::IndexedTime, x, M::Message)  =  Bridge.b(t, x, M.ℙ) + Bridge.a(t, x, M.ℙ) * r((i,t),x,M)   
Bridge.σ(t, x, M::Message) = Bridge.σ(t, x, M.ℙ)
Bridge.a(t, x, M::Message) = Bridge.a(t, x, M.ℙ)
Bridge.constdiff(M::Message) = Bridge.constdiff(M.ℙ) && Bridge.constdiff(M.ℙ̃)


function llikelihood(::LeftRule, X::SamplePath, M::Message; skip = sk)
    tt = X.tt
    xx = X.yy
    som::Float64 = 0.
    for i in 1:length(tt)-1-skip #skip last value, summing over n-1 elements
        s = tt[i]
        x = xx[i]
        r̃ = r((i,s), x, M)
        dt = tt[i+1]-tt[i]

        som += dot( Bridge._b((i,s), x, M.ℙ) - Bridge._b((i,s), x, M.ℙ̃), r̃) * dt
        if !constdiff(M)
            som -= 0.5*tr( (a((i,s), x, M.ℙ) - a((i,s), x, M.ℙ̃)) * M.H[i] )   * dt
            som += 0.5 * ( r̃' * ( a((i,s), x, M.ℙ) - a((i,s), x, M.ℙ̃) ) * r̃)  * dt
        end
    end
    som 
end

logh̃(x, h0) = -0.5 * x' * h0.H * x + h0.F' * x + h0.C    
loglik(x0, h0, Ps::Vector{PathInnovation}) = logh̃(x0, h0) + sum(getfield.(Ps,:ll))

""""
    forwardguide!((X, W, ll), (Xᵒ, Wᵒ, Wbuffer), M, ρ; skip=sk, verbose=false)

    old function in PBridge code. 
"""

function forwardguide!((X, W, ll), (Xᵒ, Wᵒ, Wbuffer), M, ρ; skip=sk, verbose=false)
    acc = false
    sample!(Wbuffer, wienertype(M.ℙ))
    Wᵒ.yy .= ρ*W.yy + sqrt(1.0-ρ^2)*Wbuffer.yy
    x0 = X.yy[1]
    solve!(Euler(),Xᵒ, x0, Wᵒ, M)
    llᵒ = llikelihood(Bridge.LeftRule(), Xᵒ, M, skip=skip)

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
    forwardguide(x0, Ms::Vector{Message}, ρs)

    Using info from Ms, and PCN-pars in ρs, starting point x0,
    forward simulate the guided process on each segment.

    On each segment a `PathInnovation`-object is constructed.
    Funtion returns a  vector of PathInnovation objects, one for each segment
"""

function forwardguide(x0, Ms::Vector{Message}, ρs)
    xend = x0
    Ps = PathInnovation[]
    for i ∈ eachindex(Ms)
        push!(Ps, PathInnovation(xend, Ms[i], ρs[i]))
        xend = lastval(Ps[i])
    end
    Ps
end


"""
    forwardguide!(::InnovationsFixed, Pᵒ::PathInnovation,  P::PathInnovation, M::Message, x0)     

    Using GuidedProposal M and innovations extracted from the W-field of P, simulate a guided process starting in x0, write into
    Pᵒ, whos `X` and `W` field are overwritten.

    Returns last value of simulated path, as also likelihood of this path
"""
function forwardguide!(::InnovationsFixed, Pᵒ::PathInnovation,  P::PathInnovation, M::Message, x0)    
    Pᵒ.W.yy .= P.W.yy
    solve!(Euler(), Pᵒ.X, x0, P.W, M)
    llᵒ = llikelihood(Bridge.LeftRule(), Pᵒ.X, M, skip=sk)
    lastval(Pᵒ), llᵒ
end

function forwardguide!(::PCN, Pᵒ::PathInnovation,  P::PathInnovation, M::Message, x0)    
    sample!(Pᵒ.Wbuf, wienertype(M.ℙ))
    ρ = Pᵒ.ρ
    Pᵒ.W.yy .= ρ * P.W.yy + sqrt(1.0-ρ^2)*Pᵒ.Wbuf.yy
    solve!(Euler(), Pᵒ.X, x0, Pᵒ.W, M)
    llᵒ = llikelihood(Bridge.LeftRule(), Pᵒ.X, M, skip=sk)
    lastval(Pᵒ), llᵒ
end

"""
    forwardguide!(gt::GuidType, Psᵒ::Vector{PathInnovation}, Ps::Vector{PathInnovation}, Ms::Vector{Message}, x0)

    Using a vector of guided process, simulate a new path on all segments. 
    The elements of Psᵒ get overwritten and hence possibly change. 
"""
function forwardguide!(gt::GuidType, Psᵒ::Vector{PathInnovation}, Ps::Vector{PathInnovation}, Ms::Vector{Message}, x0)
    x_ = x0  
    xend = 0.0*x0 ; 
    #llᴼ = 0.0 
    for i ∈ eachindex(Ps)
        xend, llᵒ = forwardguide!(gt, Psᵒ[i], Ps[i], Ms[i], x_) # profileview colours red on this line, especially when PCN is called
        x_ = xend
        ui = Psᵒ[i]
        @set! ui.ll = llᵒ
        Psᵒ[i] = ui
   end
end


