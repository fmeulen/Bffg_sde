struct ParInfo
    names::Vector{Symbol}
    recomputeguidingterm::Vector{Bool}
end
  
getfield_(P) =  (x) -> getfield(P,x)
getpar(P, ind::Vector{Symbol}) = getfield_(P).(ind)
getpar(𝒫::GuidedProcess, p::ParInfo) = getpar(𝒫.ℙ, p.names)
getpar(𝒫s::Vector{GuidedProcess}, p::ParInfo) = getpar(𝒫s[1].ℙ, p.names)

"""
    update_guidedprocess(𝒫, tup)

    Construct new instance of GuidedProcess, with fields in ℙ and ℙ̃ adjusted according to tup
    
    𝒫 = 𝒫s[3]
    tup = (C=3333333.1, A=3311.0)
    𝒫up = update_guidedprocess(𝒫,tup)
"""
function update_guidedprocess(𝒫::GuidedProcess,tup)
    # adjust ℙ
    P_ = 𝒫.ℙ
    P_ = setproperties(P_, tup)
    @set! 𝒫.ℙ = P_
    # adjust ℙ̃
    P̃_ = 𝒫.ℙ̃
    P̃_ = setproperties(P̃_, tup)
    @set! 𝒫.ℙ̃ = P̃_
    𝒫
end    


"""
    update_guidedprocesses!(𝒫s, tup)

    Construct new instance of GuidedProcess, with fields in ℙ and ℙ̃ adjusted according to tup
    Do this for each element of 𝒫s and write into it

    tup = (C=3333333.1, A=3311.0)
    update_guidedprocesses!(𝒫s,tup)
"""
function update_guidedprocesses!(𝒫s, tup)
    for i ∈ eachindex(𝒫s)
        𝒫s[i] = update_guidedprocesses(𝒫s[i], tup)
    end
end





"""
    extract parameter vector from guided process
"""
getpar(𝒫::GuidedProcess, ind::Vector{Symbol}) = getpar(𝒫.ℙ, ind::Vector{Symbol})  


say(what) = run(`osascript -e "say \"$(what)\""`, wait=false)

lastval(X::SamplePath) = X.yy[end]

lastval(ℐ::PathInnovation) = lastval(ℐ.X)

function mergepaths(ℐs)
    tt = map(x->x.X.tt, ℐs)
    yy = map(x->x.X.yy, ℐs)
    SamplePath(vcat(tt...),vcat(yy...))
end

function init_auxiliary_processes(TypeAuxProcess, obs, ℙ)
    ℙ̃s = TypeAuxProcess[]
    n = length(obs)
    for i in 1:n
      push!(ℙ̃s, TypeAuxProcess(obs[i].t, obs[i].v[1], ℙ))
    end
    ℙ̃s
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


HFC(obs::Observation) = (obs.H, obs.F, obs.C)

vectorise(P,ν, C) = vcat(SVector(P), ν, SVector(C))

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


"""
    observation_HFC(v, L, Σ)

    Convert observation v ~ N(Lx, Σ)
    to triplet  (H, F, C)
"""
function observation_HFC(v, L, Σ)
    A = L' * inv(Σ)
    A*L, A*v, logpdf(Bridge.Gaussian(zero(v), Σ), v)
end



"""
    static_accessor_HFc(u::SVector, ::Val{T}) where T
Access data stored in the container `u` so that it matches the shapes of H,F,c
and points to the correct points in `u`. `T` is the dimension of the stochastic
process.

implemented by M. Mider in GuidedProposals.jl
"""
function static_accessor_HFc(u::K, ::Val{T}) where {K<:Union{SVector,MVector},T}
    Hidx = SVector{T*T,Int64}(1:T*T)
    Fidx = SVector{T,Int64}((T*T+1):(T*T+T))
    reshape(u[Hidx], Size(T,T)), u[Fidx], u[T*T+T+1]
end


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



# plotting


ec(x,i) = getindex.(x,i)

function plot_(ℐs::Vector{PathInnovation},comp::Int)
    p = plot(ℐs[1].X.tt, ec(ℐs[1].X.yy,comp), label="")
    for k in 2:length(ℐs)
      plot!(p, ℐs[k].X.tt, ec(ℐs[k].X.yy,comp), label="")
    end
    p
end

function plot_(ℐs::Vector{PathInnovation},::String)
    p = plot(ℐs[1].X.tt, ec(ℐs[1].X.yy,2) - ec(ℐs[1].X.yy,3) , label="")
    for k in 2:length(ℐs)
      plot!(p, ℐs[k].X.tt, ec(ℐs[k].X.yy,2) - ec(ℐs[k].X.yy,3), label="")
    end
    p
end



function plot_all(ℐs::Vector{PathInnovation})
    p1 = plot_(ℐs,1)
    p2 = plot_(ℐs,2)
    p3 = plot_(ℐs,3)
    p4 = plot_(ℐs,4)
    p5 = plot_(ℐs,5)
    p6 = plot_(ℐs,6)
    p2_3 = plot_(ℐs,"23")
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
