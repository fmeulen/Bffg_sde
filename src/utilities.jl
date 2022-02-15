function printinfo(ll, llᵒ, s::String) 
    println(s * " update. ll $ll; llᵒ $llᵒ, difference: ",round(llᵒ-ll;digits=3)) 
    #println()
end

say(what) = run(`osascript -e "say \"$(what)\""`, wait=false)

τ(S,T) = (x) ->  S + (x-S) * (2-(x-S)/(T-S))

timegrid(S, T, M) = τ(S,T).(collect(range(S, T, length=M)))

function set_timegrids(obs, dt)  
    i=1
    M = Int64(ceil((obs[i+1].t-obs[i].t)/dt))
    out = [timegrid(obs[i].t, obs[i+1].t, M)]
    for i ∈ 2:length(obs)-1
        M = Int64(ceil((obs[i+1].t-obs[i].t)/dt))
        push!(out, timegrid(obs[i].t, obs[i+1].t, M))
    end
    out
end

function convert_PνC_to_HFC(P,ν,C)
    H = inv(P)
    Htransform(H, P\ν, C)
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








