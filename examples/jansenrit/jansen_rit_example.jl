using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
using ForwardDiff
using DifferentialEquations
using Setfield
using Plots

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
#θtrue =[3.25, 100.0, 22.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 0.0, 2000.0]
ℙ = JansenRitDiffusion(θtrue...)
T = 1.0
ℙ̃ = JansenRitDiffusionAux(ℙ.a, ℙ.b , ℙ.A , ℙ.μy, ℙ.σy, T)


#---- generate test data
x0 = @SVector [0.08, 18.0, 15.0, -0.5, 0.0, 0.0] 
W = sample((-1.0):0.0001:T, Wiener())                        #  sample(tt, Wiener{ℝ{1}}())
Xf_prelim = solve(Euler(), x0, W, ℙ)
# drop initial nonstationary behaviour
Xf = SamplePath(Xf_prelim.tt[10001:end], Xf_prelim.yy[10001:end])
x0 = Xf.yy[1]


#------  set observations
L = @SMatrix [0.0 1.0 -1.0 0.0 0.0 0.0]
m,  = size(L)
Σdiagel = 10e-5
Σ = SMatrix{m,m}(Σdiagel*I)

skipobs = 100#  length(Xf.tt)-1 #500
obstimes =  Xf.tt[1:skipobs:end]
obsvals = map(x -> L*x, Xf.yy[1:skipobs:end])
plot_all(Xf)
savefig("forwardsimulated.png")



#------- process observations
obs = Observation[]
for i ∈ eachindex(obsvals)
    push!(obs, Observation(obstimes[i], obsvals[i], L, Σ))
end
timegrids = set_timegrids(obs, 0.00005)
ρ = 0.95
ρs = fill(ρ, length(timegrids))
#------- Backwards filtering
(H0, F0, C0), 𝒫s = backwardfiltering(obs, timegrids, ℙ, ℙ̃);

# Forwards guiding initialisation
ℐs, ll = forwardguide(x0, 𝒫s, ρs);
plot_all(ℐs)
savefig("guidedinitial.png")


# check whether interpolation goes fine
 
deviations = [ obs[i].v - obs[i].L * lastval(ℐs[i-1])  for i in 2:length(obs)]
#plot(obstimes[2:end], map(x-> x[1,1], deviations))

    
# Forwards guiding pCN

# settings sampler
iterations = 30 # 5*10^4
skip_it = 10  #1000
subsamples = 0:skip_it:iterations

XX = Any[]
(0 in subsamples) &&    push!(XX, mergepaths(ℐs))

ℙinit = ℙ # @set ℙ.A=100.0
ℙ̃init = ℙ̃ # @set ℙ̃.A=50.0


ℐs, ll = forwardguide(x0, 𝒫s, ρs);
ℐsᵒ, llᵒ = forwardguide(x0, 𝒫s, ρs);#deepcopy(ℐs)
verbose = false

#@enter forwardguide!(PCN(), ℐsᵒ, ℐs, 𝒫s, x0);

for iter in 1:iterations
  logh0, llᵒ = forwardguide!(PCN(), ℐsᵒ, ℐs, 𝒫s, x0);
  llᵒ = logh0 + llᵒ
  # println(llᵒ)
  # println(lastval(ℐsᵒ[3]))
  # println(lastval(ℐs[3]))
  
  
  dll = llᵒ - ll
  !verbose && print("ll $ll $llᵒ, diff_ll: ",round(dll;digits=3)) 

  if log(rand()) < dll 
     #ℐs .= ℐsᵒ
     ℐs, ℐsᵒ = ℐsᵒ,  ℐs
     ll = llᵒ

    #  for i in eachindex(ℐs)
    #   ℐs[i] = ℐsᵒ[i]
    #  end


  #   println(ℐs == ℐsᵒ)
    !verbose && print("✓")    
  end 
  println()

  (iter in subsamples) && push!(XX, mergepaths(ℐs))
end






pℐ =  plot_all(ℐs)
pXf = plot_all(Xf)
plot(pℐ, pXf)
savefig("guidedinitial_onepCNstep.png")



#---------------------- a program




ℐs = forwardguide(x0, 𝒫s, ρs)
plot_all(ℐs)
savefig("guidedinitial.png")



# forwardguide!(PCN(), ℐs, 𝒫s, x0);
#  ℐ, 𝒫 =  ℐs[end], 𝒫s[end];
#  va = checkcorrespondence(ℐ, 𝒫)

#  forwardguide!(InnovationsFixed(), ℐs, 𝒫s, x0; skip=sk, verbose=true);
#  ℐ, 𝒫 =  ℐs[end], 𝒫s[end]
#  va = checkcorrespondence(ℐ, 𝒫)


ℐs = forwardguide(x0, 𝒫s, ρs)

@enter  forwardguide!(PCN(), ℐs, 𝒫s, x0,  verbose=false);

@enter checkcorrespondence(ℐ, 𝒫)

for i in eachindex(ℐs)
  checkcorrespondence(ℐs[i], 𝒫s[i])
end


𝒫sᵒ = deepcopy(𝒫s)
ℐsᵒ = deepcopy(ℐs) # need to create only once
θθ =[getpar(𝒫s[1].ℙ)]

# estimate (C)
tp = [20.0] # 20.0*[0.1 0.0; 0.0 0.1]

acc = 0
for iter in 1:iterations
    global acc, ℐs, 𝒫s, ℐsᵒ, 𝒫sᵒ
    a = forwardguide!(PCN(), ℐs, 𝒫s, x0,  verbose=false);
    #a = forwardguide!(InnovationsFixed(), ℐs, 𝒫s, x0,  verbose=true);
    ℐ, 𝒫 = ℐs[end], 𝒫s[end]
    checkcorrespondence(ℐ, 𝒫)

    acc += a
  #  (iter in subsamples) && push!(XX, mergepaths(ℐs))    #  or use copy(X)  ?
    println(iter)


    if iter>5
  #   (θ, accθ) = parupdate!(obs, timegrids, x0, (𝒫s, ℐs), (𝒫sᵒ, ℐsᵒ); tuningpars = tp)
    if iter==500
        #tp = cov(hcat(ec(θθ,1), ec(θθ,2))) * (2.38)^2/6.0
    end

    #println(accθ)
    #push!(θθ, θ)
    end
end




#say("Joehoe, klaar met rekenen")

plot_all(ℐs)
savefig("guidedfinal.png")


pth1 = plot(ec(θθ,1))
# pth2 = plot(ec(θθ,2))
# plot(pth1, pth2, layout= (@layout [a b]))
savefig("thetas.png")

println(θθ)










# priorθ = Dict(:A => Uniform(0.0, 20.0), 
# 			  :B => Uniform(0.0, 50.0), 
# 			  :C => TruncatedNormal(100,50,0.0,Inf64), 
# 			  :μy=> Normal(0.0, 10.0^6), 
# 			  :σy => Uniform(10.0, 5000.0))











if false 

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


end













