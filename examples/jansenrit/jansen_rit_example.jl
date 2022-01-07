using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
using ForwardDiff
using DifferentialEquations


import Bridge: R3, IndexedTime, llikelihood, kernelr3, constdiff, Euler, solve, solve!
import ForwardDiff: jacobian

wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")
include("jansenrit.jl")

include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/funcdefs.jl")


################################  TESTING  ################################################
# settings sampler
iterations = 3_000 # 5*10^4
skip_it = 100  #1000
subsamples = 0:skip_it:iterations

T = 10.0

sk = 0 # skipped in evaluating loglikelihood

θtrue =[3.25, 100.0, 22.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 2000.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
# θtrue =[0.0, 100.0, 0.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 00.0, 2000.0]  
# θtrue =[3.25, 1.0, 22.0, 0.5, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 20.0]  # adjust a and b

ℙ = JansenRitDiffusion(θtrue...)
x0 = @SVector [0.08, 18.0, 15.0, -0.5, 0.0, 0.0] 
#x0 = @SVector zeros(dim(ℙ))
ℙ̃ = JansenRitDiffusionAux(ℙ.a, ℙ.b , ℙ.A , ℙ.μy, ℙ.σy, T)

#---- generate test data
Random.seed!(4)
W = sample((-1.0):0.001:T, Wiener())                        #  sample(tt, Wiener{ℝ{1}}())
Xf_prelim = solve(Euler(), x0, W, ℙ)
# drop initial nonstationary behaviour
Xf = SamplePath(Xf_prelim.tt[1001:end], Xf_prelim.yy[1001:end])
x0 = Xf.yy[1]
using Plots
k = 3; plot(Xf.tt, getindex.(Xf.yy,k))


#------  set observations
L = @SMatrix [0.0 1.0 -1.0 0.0 0.0 0.0]
m,  = size(L)
Σdiagel = 10e-9
Σ = SMatrix{m,m}(Σdiagel*I)

obstimes = Xf.tt[1:100:end]
obsvals = map(x -> L*x, Xf.yy[1:100:end])
obs = Observation[]
for i ∈ eachindex(obsvals)
    push!(obs, Observation(obstimes[i], obsvals[i], L, Σ))
end




# Backwards filtering
  
function backwardfiltering(obs, ℙ, ℙ̃ ;ϵ = 10e-2, M=100)
    Hinit, Finit, Cinit =  init_HFC(obs[end].v, obs[end].L, dim(ℙ); ϵ=ϵ)
    n = length(obs)

    HT, FT, CT = fusion_HFC(HFC(obs[n]), (Hinit, Finit, Cinit) )
    𝒫s = GuidedProcess[]
    for i in n:-1:2
        println(i)
        tt = timegrid(obs[i-1].t, obs[i].t, M=M)
        𝒫 = GuidedProcess(DE(Vern7()), ℙ, ℙ̃, tt, HT, FT, CT)
        pushfirst!(𝒫s, 𝒫)
        message = (𝒫.H[1], 𝒫.F[1], 𝒫.C[1])
        (HT, FT, CT) = fusion_HFC(message, HFC(obs[i-1]))
    end
    (HT, FT, CT), 𝒫s
end

backwardfiltering(obs, ℙ, ℙ̃)


# Forwards guiding initialisation
xend = x0
ℐs = PathInnovation[]
for i ∈ 1:n-1
    push!(ℐs, PathInnovation(xend, 𝒫s[i]))
    xend = lastval(ℐs[i])
end


# plotting and checking
        ec(x,i) = getindex.(x,i)

        p = plot(ℐs[1].X.tt, ec(ℐs[1].X.yy,1), label="")
        for k in 2:n-1
        plot!(p, ℐs[k].X.tt, ec(ℐs[k].X.yy,1), label="")
        end
        p

        # check whether interpolation goes fine
        for i in 2:n-1
        println( obs[i+1].v - obs[i].L * ℐs[i].X.yy[end]  )
        end

# Forwards guiding pCN
xend = x0  
for i ∈ 1:n-1
    (ℐs[i], xend, acc) = forwardguide(xend, ℐs[i], 𝒫s[i], ρ);
end




# further initialisation
XX = Any[]
if 0 in subsamples
    push!(XX, copy(X))
end


ρ = .9  # 0.99999999


iterations = 10
acc = 0
for iter in 1:iterations
    global acc

    # Forwards guiding pCN
    xend = x0  
    for i ∈ 1:n-1
        (ℐs[i], xend, a) = forwardguide(xend, ℐs[i], 𝒫s[i], ρ);
    end

    
    
    if iter in subsamples
    #    push!(XX, copy(X))
        push!(XX, mergepaths(ℐs))
    end
    acc += a

end

@info "Done."*"\x7"^6

include("process_output.jl")









####################### MH algorithm ###################
dt = 1/500
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

W = sample(tt, wienertype(𝒫.ℙ))    #W = sample(tt, Wiener())
X = solve(Euler(), x0, W, ℙ)  # allocation
solve!(Euler(),X, x0, W, 𝒫)
Xᵒ = deepcopy(X)
ll = llikelihood(Bridge.LeftRule(), X, 𝒫, skip=sk)
Wᵒ = deepcopy(W)
Wbuffer = deepcopy(W)




if false 
    using Plots
    l = @layout [a b c ; d e f]
    p1 = plot(X.tt, getindex.(X.yy,1))
    plot!(p1, Xf.tt, getindex.(Xf.yy,1))
    p2 = plot(X.tt, getindex.(X.yy,2))
    plot!(p2, Xf.tt, getindex.(Xf.yy,2))
    p3 = plot(X.tt, getindex.(X.yy,3))
    plot!(p3, Xf.tt, getindex.(Xf.yy,3))
    p4 = plot(X.tt, getindex.(X.yy,4))
    plot!(p4, Xf.tt, getindex.(Xf.yy,4))
    p5 = plot(X.tt, getindex.(X.yy,5))
    plot!(p5, Xf.tt, getindex.(Xf.yy,5))
    p6 = plot(X.tt, getindex.(X.yy,6))
    plot!(p6, Xf.tt, getindex.(Xf.yy,6))
    plot(p1,p2,p3,p4,p5,p6, layout=l)

    LT*X.yy[end] - vT

    p = plot(X.tt, getindex.(X.yy,2) - getindex.(X.yy,3))
    plot!(p, Xf.tt, getindex.(Xf.yy,2) - getindex.(Xf.yy,3))
end

# further initialisation
XX = Any[]
if 0 in subsamples
    push!(XX, copy(X))
end


ρ = .9  # 0.99999999


acc = 0
for iter in 1:iterations
    global acc
    (X, W, ll), a = forwardguide!((X, W, ll), (Xᵒ, Wᵒ, Wbuffer), 𝒫, ρ; skip=sk, verbose=false)
    if iter in subsamples
        push!(XX, copy(X))
    end
    acc += a

end

@info "Done."*"\x7"^6








# multiple time intervals



