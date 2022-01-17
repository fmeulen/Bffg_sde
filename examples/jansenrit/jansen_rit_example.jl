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
using RCall

import Bridge: R3, IndexedTime, llikelihood, kernelr3, constdiff, Euler, solve, solve!
import ForwardDiff: jacobian


wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")
include("jansenrit.jl")
include("jansenrit3.jl")

include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/funcdefs.jl")
include("/Users/frankvandermeulen/.julia/dev/Bffg_sde/src/utilities.jl")

################################  TESTING  ################################################

sk = 0 # skipped in evaluating loglikelihood


Random.seed!(5)

model= [:jr, :jr3][2]

if model == :jr
  θtrue =[3.25, 100.0, 22.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 2000.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
  ℙ = JansenRitDiffusion(θtrue...)
  AuxType = JansenRitDiffusionAux
end
if model == :jr3
  θtrue =[3.25, 100.0, 22.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 0.1, 2000.0, 1.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
  ℙ = JansenRitDiffusion3(θtrue...)
  AuxType = JansenRitDiffusionAux3
end


#---- generate test data
T = 1.0
#x0 = @SVector [0.08, 18.0, 15.0, -0.5, 0.0, 0.0] 
x0 = @SVector zeros(6)
W = sample((-1.0):0.0001:T, wienertype(ℙ))                        #  sample(tt, Wiener{ℝ{1}}())
Xf_prelim = solve(Euler(), x0, W, ℙ)
# drop initial nonstationary behaviour
Xf = SamplePath(Xf_prelim.tt[10001:end], Xf_prelim.yy[10001:end])
x0 = Xf.yy[1]


#------  set observations
L = @SMatrix [0.0 1.0 -1.0 0.0 0.0 0.0]
m,  = size(L)
Σdiagel = 1e-7
Σ = SMatrix{m,m}(Σdiagel*I)

skipobs = 100#  length(Xf.tt)-1 #500
obstimes =  Xf.tt[1:skipobs:end]
obsvals = map(x -> L*x, Xf.yy[1:skipobs:end])
pF = plot_all(Xf, obstimes, obsvals)
savefig("forwardsimulated.png")



#------- process observations
obs = Observation[]
for i ∈ eachindex(obsvals)
    push!(obs, Observation(obstimes[i], obsvals[i], L, Σ))
end
obs[1] = Observation(obstimes[1], x0, SMatrix{6,6}(1.0I), SMatrix{6,6}(Σdiagel*I))

timegrids = set_timegrids(obs, 0.00005)
ρ = 0.95
ρs = fill(ρ, length(timegrids))
#------- Backwards filtering

ℙ̃s = init_auxiliary_processes(AuxType, obs, ℙ)
(H0, F0, C0), 𝒫s = backwardfiltering(obs, timegrids, ℙ, ℙ̃s);


# Forwards guiding initialisation
ℐs = forwardguide(x0, 𝒫s, ρs);
plot_all(ℐs)
savefig("guidedinitial.png")

pf = plot_all(Xf)
pg = plot_all(ℐs)
l = @layout  [a;b]
plot(pf, pg, layout=l)
savefig("forward_and_guidedinital.png")

plot(map(x->x.ll, ℐs))

# check whether interpolation goes fine
 
deviations = [ obs[i].v - obs[i].L * lastval(ℐs[i-1])  for i in 2:length(obs)]
#plot(obstimes[2:end], map(x-> x[1,1], deviations))

ρ = 0.98
tp = [5.0]
ℙinit = ℙ #  @set ℙ.C=280.0
#ℙ̃init = ℙ̃ # @set ℙ̃.A=50.0


ℙ̃s_init = init_auxiliary_processes(AuxType, obs, ℙinit)



XX, θs, ℐs, (accpar, accinnov) =   parinf(obs, timegrids, x0, tp, ρ, ℙinit, ℙ̃s_init; 
                skip_it = 100, iterations=4_000, verbose=true, parupdating=true);    


pℐ =  plot_all(ℐs)

l = @layout [a ;b]
plot(pF, pℐ,  layout=l)
savefig("forward_and_guided_lastiterate.png")

pθ = plot(map(x->x[1], θs), label="θ")
savefig("thetas.png")

p23 = plot_(ℐs,"23")
plot!(p23, Xf.tt, getindex.(Xf.yy,2) - getindex.(Xf.yy,3), label="")
savefig("second_minus_third.png")


PLOT = true

if PLOT

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













