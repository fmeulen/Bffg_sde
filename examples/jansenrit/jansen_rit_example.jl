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
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/utilities.jl")

################################  TESTING  ################################################

sk = 0 # skipped in evaluating loglikelihood
ρ = 0.9

Random.seed!(5)



θtrue =[3.25, 100.0, 22.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 2000.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
# θtrue =[0.0, 100.0, 0.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 00.0, 2000.0]  
# θtrue =[3.25, 1.0, 22.0, 0.5, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 20.0]  # adjust a and b

ℙ = JansenRitDiffusion(θtrue...)
T = 10.0
ℙ̃ = JansenRitDiffusionAux(ℙ.a, ℙ.b , ℙ.A , ℙ.μy, ℙ.σy, T)


#---- generate test data
x0 = @SVector [0.08, 18.0, 15.0, -0.5, 0.0, 0.0] 
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
@time (H0, F0, C0), 𝒫s = backwardfiltering(obs, ℙ, ℙ̃);

# Forwards guiding initialisation
n = length(obs)
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
ℐs, acc = forwardguide(x0, ℐs, 𝒫s, ρ);



# settings sampler
iterations = 25 # 5*10^4
skip_it = 10  #1000
subsamples = 0:skip_it:iterations

XX = Any[]
(0 in subsamples) &&    push!(XX, mergepaths(ℐs))


acc = 0
for iter in 1:iterations
    global acc
    ℐs, a = forwardguide(x0, ℐs, 𝒫s, ρ);
    acc += a
    (iter in subsamples) && push!(XX, mergepaths(ℐs))    #    push!(XX, copy(X))
end

say("Joehoe, klaar met rekenen")


#--------- plotting 
extractcomp(v,i) = map(x->x[i], v)
d = dim(ℙ)
J = length(XX[1].tt)
iterates = [Any[s, XX[i].tt[j], k, XX[i].yy[j][k]] for k in 1:d, j in 1:J, (i,s) in enumerate(subsamples) ][:]
# FIXME, J need not be constant


df_iterates = DataFrame(iteration=extractcomp(iterates,1),time=extractcomp(iterates,2), component=extractcomp(iterates,3), value=extractcomp(iterates,4))
#CSV.write(outdir*"iterates.csv",df_iterates)


################ plotting in R ############
using RCall
dd = df_iterates

@rput dd
#@rput obs_scheme
@rput outdir

R"""
library(ggplot2)
library(tidyverse)
theme_set(theme_bw(base_size = 13))

# vT = c(0.03125,   0.25,   1.0)                  #vT <- c(5/128,3/8,2)
# vTvec = rep(vT, nrow(dd)/3)

dd$component <- as.factor(dd$component)
dd <- dd %>% mutate(component=fct_recode(component,'component 1'='1',
              'component 2'='2', 'component 3'='3', 'component 4'='4','component 5'='5','component 6'='6'))




# make figure
p <- ggplot(mapping=aes(x=time,y=value,colour=iteration),data=dd) +
  geom_path(aes(group=iteration)) + #geom_hline(aes(yintercept=trueval)) +
  facet_wrap(~component,scales='free_y')+
  scale_colour_gradient(low='green',high='blue')+ylab("")
show(p)

# write to pdf
fn <- paste0(outdir,"bridges.pdf")
pdf(fn,width=7,height=5)
show(p)
dev.off()    

"""



if false 








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


end

