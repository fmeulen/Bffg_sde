
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




# PLOT = false 

# if PLOT
# using RCall
# #--------- plotting 
# extractcomp(v,i) = map(x->x[i], v)
# d = dim(ℙ)
# J = length(XX[1].tt)
# iterates = [Any[s, XXsave[i].tt[j], k, XXsave[i].yy[j][k]] for k in 1:d, j in 1:J, (i,s) in enumerate(subsamples) ][:]
# # FIXME, J need not be constant


# df_iterates = DataFrame(iteration=extractcomp(iterates,1),time=extractcomp(iterates,2), component=extractcomp(iterates,3), value=extractcomp(iterates,4))
# CSV.write(outdir*"iterates.csv",df_iterates)

################ plotting in R ############

# #using RCall
# dd = df_iterates

# @rput dd
# #@rput obs_scheme
# @rput outdir

# R"""
# library(ggplot2)
# library(tidyverse)
# theme_set(theme_bw(base_size = 13))

# dd$component <- as.factor(dd$component)
# dd <- dd %>% mutate(component=fct_recode(component,'component 1'='1',
#               'component 2'='2', 'component 3'='3', 'component 4'='4','component 5'='5','component 6'='6'))

# # make figure
# p <- ggplot(mapping=aes(x=time,y=value,colour=iteration),data=dd) +
#   geom_path(aes(group=iteration)) + #geom_hline(aes(yintercept=trueval)) +
#   facet_wrap(~component,scales='free_y')+
#   scale_colour_gradient(low='green',high='blue')+ylab("")
# show(p)

# # write to pdf
# fn <- paste0(outdir,"bridges.pdf")
# pdf(fn,width=7,height=5)
# show(p)
# dev.off()    

# """


# end













