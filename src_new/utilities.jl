
say(what) = run(`osascript -e "say \"$(what)\""`, wait=false)

τ(S,T) = (x) ->  S + (x-S) * (2-(x-S)/(T-S))

timegrid(S, T, M) = τ(S,T).(collect(range(S, T, length=M)))

function set_timegrids(obs, dt)  
    out=Vector{Float64}[]
    for i ∈ 1:length(obs)-1
        M = Int64(ceil((obs[i+1].t-obs[i].t)/dt))
        push!(out, timegrid(obs[i].t, obs[i+1].t, M))
    end
    out
end



getfield_(P) =  (x) -> getfield(P,x)
getpar(P, ind::Vector{Symbol}) = getfield_(P).(ind)
getpar(M::Message, p::ParInfo) = getpar(M.ℙ, p.names)
getpar(Ms::Vector{Message}, p::ParInfo) = getpar(Ms[1].ℙ, p.names)


function convert_PνC_to_HFC(P,ν,C)
    H = inv(P)
    Htransform(H, P\ν, C)
end   



"""
    extract parameter vector from guided process
"""
getpar(M::Message, ind::Vector{Symbol}) = getpar(M.ℙ, ind::Vector{Symbol})  




lastval(X::SamplePath) = X.yy[end]

lastval(P::Union{PathInnovation, PathInnovationProposal}) = lastval(P.X)

function mergepaths(Ps)
    tt = map(x->x.X.tt, Ps)
    yy = map(x->x.X.yy, Ps)
    SamplePath(vcat(tt...),vcat(yy...))
end



   






"""
    init_HFC(v, L, d; ϵ=0.01)

    d = dimension of the diffusion
    First computes xT = L^(-1) * vT (Moore-Penrose inverse), a reasonable guess for the full state based on the partial observation vT
    Then convert artifical observation v ~ N(xT, ϵ^(-1) * I)
    to triplet  (H, F, C)
"""
function init_HFC(v, L, d::Int64; ϵ=0.01)
    P = ϵ^(-1)*SMatrix{d,d}(1.0I)
    xT = L\v
    z = zero(xT)
    C = logpdf(Bridge.Gaussian(z, P), z) 
    convert_PνC_to_HFC(P, xT ,C)
end








