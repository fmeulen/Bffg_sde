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


say(what) = run(`osascript -e "say \"$(what)\""`, wait=false)

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









# plotting


ec(x,i) = getindex.(x,i)


function plot_all(::JansenRitDiffusion, X::SamplePath)
    p1 = plot(X.tt, getindex.(X.yy,1), label="")
    p2 = plot(X.tt, getindex.(X.yy,2), label="")
    p3 = plot(X.tt, getindex.(X.yy,3), label="")
    p4 = plot(X.tt, getindex.(X.yy,4), label="")
    p5 = plot(X.tt, getindex.(X.yy,5), label="")
    p6 = plot(X.tt, getindex.(X.yy,6), label="")
    p2_3 = plot(X.tt, getindex.(X.yy,2) - getindex.(X.yy,3), label="")
    l = @layout [a b c; d e f; g]
    plot(p1,p2,p3,p4,p5,p6, p2_3, layout=l)
end

function plot_all(::JansenRitDiffusion,X::SamplePath, obstimes, obsvals)
    p1 = plot(X.tt, getindex.(X.yy,1), label="")
    p2 = plot(X.tt, getindex.(X.yy,2), label="")
    p3 = plot(X.tt, getindex.(X.yy,3), label="")
    p4 = plot(X.tt, getindex.(X.yy,4), label="")
    p5 = plot(X.tt, getindex.(X.yy,5), label="")
    p6 = plot(X.tt, getindex.(X.yy,6), label="")
    p2_3 = plot(X.tt, getindex.(X.yy,2) - getindex.(X.yy,3), label="")
    plot!(p2_3, obstimes, map(x->x[1], obsvals), seriestype=:scatter, markersize=1.5, label="")
    l = @layout [a b c; d e f; g]
    plot(p1,p2,p3,p4,p5,p6, p2_3, layout=l)
end



function plotboth(::JansenRitDiffusion,X::SamplePath, timegrids, XX, comp)
    p1 = plot(X.tt, getindex.(X.yy,comp), label="",color="grey")
    for k in eachindex(XX)
        plot!(p1, timegrids[k], ec(XX[k],comp), label="")
    end
    p1
end

function plot_all(P::JansenRitDiffusion,X::SamplePath, obstimes, obsvals, timegrids, XX)
    p1 = plotboth(P, X, timegrids, XX, 1)
    p2 = plotboth(P, X, timegrids, XX, 2)
    p3 = plotboth(P, X, timegrids, XX, 3)
    p4 = plotboth(P, X, timegrids, XX, 4)
    p5 = plotboth(P, X, timegrids, XX, 5)
    p6 = plotboth(P, X, timegrids, XX, 6)
    p2_3 =  plot(X.tt, getindex.(X.yy,2)-getindex.(X.yy,3), label="",color="grey")
    for k in eachindex(XX)
        plot!(p2_3, timegrids[k], ec(XX[k],2)-ec(XX[k],3), label="")
    end
    p1
    plot!(p2_3, obstimes, map(x->x[1], obsvals), seriestype=:scatter, markersize=1.5, label="")
    l = @layout [a b c; d e f; g]
    plot(p1,p2,p3,p4,p5,p6, p2_3, layout=l)
end


function plot_(::JansenRitDiffusion, tt, XX, comp::Int)
    p = plot(tt[1], ec(XX[1],comp), label="")
    for k in 2:length(XX)
      plot!(p, tt[k], ec(XX[k],comp), label="")
    end
    p
end


function plot_all(P::JansenRitDiffusion, timegrids, XX)
    tt = timegrids
    p1 = plot_(P, tt, XX, 1)
    p2 = plot_(P, tt, XX, 2)
    p3 = plot_(P, tt, XX, 3)
    p4 = plot_(P, tt, XX, 4)
    p5 = plot_(P, tt, XX, 5)
    p6 = plot_(P, tt, XX, 6)
    p7 = plot(tt[1], ec(XX[1],2) - ec(XX[1],3), label="")
    for k in 2:length(XX)
      plot!(p7, tt[k], ec(XX[k],2) - ec(XX[k],3), label="")
    end
  
    l = @layout [a b c ; d e f; g]
    plot(p1, p2, p3, p4, p5, p6, p7, layout=l)
end