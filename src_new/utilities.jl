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

function plot_(Ps::Vector,comp::Int)
    p = plot(Ps[1].X.tt, ec(Ps[1].X.yy,comp), label="")
    for k in 2:length(Ps)
      plot!(p, Ps[k].X.tt, ec(Ps[k].X.yy,comp), label="")
    end
    p
end

function plot_(Ps,::String)
    p = plot(Ps[1].X.tt, ec(Ps[1].X.yy,2) - ec(Ps[1].X.yy,3) , label="")
    for k in 2:length(Ps)
      plot!(p, Ps[k].X.tt, ec(Ps[k].X.yy,2) - ec(Ps[k].X.yy,3), label="")
    end
    p
end



function plot_all(Ps::Vector)
    p1 = plot_(Ps,1)
    p2 = plot_(Ps,2)
    p3 = plot_(Ps,3)
    p4 = plot_(Ps,4)
    p5 = plot_(Ps,5)
    p6 = plot_(Ps,6)
    p2_3 = plot_(Ps,"23")
    l = @layout [a b c ; d e f; g]
    plot(p1, p2, p3, p4, p5, p6, p2_3, layout=l)
end

function plot_all(X::SamplePath)
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

function plot_all(X::SamplePath, obstimes, obsvals)
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



function plotboth(X::SamplePath, Ps, comp)
    p1 = plot(X.tt, getindex.(X.yy,comp), label="",color="grey")
    for k in 1:length(Ps)
        plot!(p1, Ps[k].X.tt, ec(Ps[k].X.yy,comp), label="")
    end
    p1
end

function plot_all(X::SamplePath, obstimes, obsvals,Ps)
    p1 = plotboth(X, Ps, 1)
    p2 = plotboth(X, Ps, 2)
    p3 = plotboth(X, Ps, 3)
    p4 = plotboth(X, Ps, 4)
    p5 = plotboth(X, Ps, 5)
    p6 = plotboth(X, Ps, 6)
    
    p2_3 = plot_(Ps,"23")
    p2_3 = plot!(X.tt, getindex.(X.yy,2) - getindex.(X.yy,3), label="", color="grey")
    plot!(p2_3, obstimes, map(x->x[1], obsvals), seriestype=:scatter, markersize=1.5, label="")
    

    l = @layout [a b c; d e f; g]
    plot(p1,p2,p3,p4,p5,p6, p2_3, layout=l)
end


# X = Xf1
# p1 = plot(X.tt, getindex.(X.yy,1), label="")
# p2 = plot(X.tt, getindex.(X.yy,2), label="")
# p3 = plot(X.tt, getindex.(X.yy,3), label="")
# p4 = plot(X.tt, getindex.(X.yy,4), label="")
# p5 = plot(X.tt, getindex.(X.yy,5), label="")
# p6 = plot(X.tt, getindex.(X.yy,6), label="")
# X = Xf2
# plot!(p1, X.tt, getindex.(X.yy,1), label="")
# plot!(p2, X.tt, getindex.(X.yy,2), label="")
# plot!(p3, X.tt, getindex.(X.yy,3), label="")
# plot!(p4, X.tt, getindex.(X.yy,4), label="")
# plot!(p5, X.tt, getindex.(X.yy,5), label="")
# plot!(p6, X.tt, getindex.(X.yy,6), label="")





# p2_3 = plot(X.tt, getindex.(X.yy,2) - getindex.(X.yy,3), label="")
# plot!(p2_3, obstimes, map(x->x[1], obsvals), seriestype=:scatter, markersize=1.5, label="")
# l = @layout [a b c; d e f]
# plot(p1,p2,p3,p4,p5,p6, layout=l)


function plot_all(XX)
    XXcat = vcat(XX...)
    tt = vcat(timegrids...)
    p1 = plot(tt, getindex.(XXcat,1), label="")
    p2 = plot(tt, getindex.(XXcat,2), label="")
    p3 = plot(tt, getindex.(XXcat,3), label="")
    p4 = plot(tt, getindex.(XXcat,4), label="")
    p5 = plot(tt, getindex.(XXcat,5), label="")
    p6 = plot(tt, getindex.(XXcat,6), label="")
    p7 = plot(tt, getindex.(XXcat,2)-getindex.(XXcat,3), label="")
    l = @layout [a b c ; d e f; g]
    plot(p1, p2, p3, p4, p5, p6, p7, layout=l)
end