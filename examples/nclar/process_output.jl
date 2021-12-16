# write mcmc iterates to csv file
extractcomp(v,i) = map(x->x[i], v)
d = dim(ℙ)
iterates = [Any[s, tt[j], k, XX[i].yy[j][k]] for k in 1:d, j in 1:length(X), (i,s) in enumerate(subsamples) ][:]
df_iterates = DataFrame(iteration=extractcomp(iterates,1),time=extractcomp(iterates,2), component=extractcomp(iterates,3), value=extractcomp(iterates,4))
CSV.write(outdir*"iterates-"*obs_scheme*".csv",df_iterates)


################ plotting in R ############
using RCall
dd = df_iterates

@rput dd
@rput obs_scheme
@rput outdir

R"""
library(ggplot2)
library(tidyverse)
theme_set(theme_bw(base_size = 13))

vT = c(0.03125,   0.25,   1.0)                  #vT <- c(5/128,3/8,2)
vTvec = rep(vT, nrow(dd)/3)

dd$component <- as.factor(dd$component)
dd <- dd %>% mutate(component=fct_recode(component,'component 1'='1','component 2'='2','component 3'='3'))%>% mutate(trueval = vTvec)

# make figure
p <- ggplot(mapping=aes(x=time,y=value,colour=iteration),data=dd) +
  geom_path(aes(group=iteration)) + geom_hline(aes(yintercept=trueval)) +
  facet_wrap(~component,ncol=1,scales='free_y')+
  scale_colour_gradient(low='green',high='blue')+ylab("")
show(p)

# write to pdf
fn <- paste0(outdir,obs_scheme,".pdf")
pdf(fn,width=7,height=5)
show(p)
dev.off()    

"""


################ write settings to file ############

ave_acc_perc = 100*round(acc/iterations;digits=2)

fn = outdir*"info-"*obs_scheme*".txt"
f = open(fn,"w")
write(f, "Choice of observation schemes: ",obs_scheme,"\n")
write(f, "Easy conditioning (means going up to 1 for the rough component instead of 2): ",string(easy_conditioning),"\n")
write(f, "Number of iterations: ",string(iterations),"\n")
write(f, "Skip every ",string(skip_it)," iterations, when saving to csv","\n\n")
write(f, "Starting point: ",string(x0),"\n")
write(f, "End time T: ", string(T),"\n")
write(f, "Endpoint v: ",string(vT),"\n")
write(f, "Noise Sigma: ",string(ΣT),"\n")
write(f, "L: ",string(LT),"\n\n")
write(f,"Mesh width: ",string(dt),"\n")
write(f, "rho (Crank-Nicholsen parameter: ",string(ρ),"\n")
write(f, "Average acceptance percentage: ",string(ave_acc_perc),"\n\n")
#write(f, "Backward type parametrisation in terms of nu and H? ",string(νHparam),"\n")
close(f)


println("Average acceptance percentage: ",ave_acc_perc,"\n")
println(obs_scheme)




