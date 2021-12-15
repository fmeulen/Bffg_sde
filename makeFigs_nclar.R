  outdir_nclar <- "/Users/frankvandermeulen/Sync/DOCUMENTS/onderzoek/code/diffbridges/out_nclar/"
  
  library(ggplot2)
  library(ggthemes)
  library(tidyverse)
  library(gridExtra)
  
  theme_set(theme_minimal())
  
  obs_scheme <- c("full", "firstcomponent")[1]
  vT = c(0.03125,   0.25,   1.0)
  #vT <- c(5/128,3/8,2)
  
  # read iterates
  fn <- paste0(outdir_nclar,"iterates-",obs_scheme,".csv")
  print(fn)
  d <- read.csv(fn)
  d$component <- as.factor(d$component)
  d <- d %>% mutate(component=fct_recode(component,'component 1'='1','component 2'='2','component 3'='3'))
  vTvec = rep(vT, nrow(d)/3)
  d <- d %>% mutate(trueval = vTvec)
  
  
  # make figure
  fn <- paste0(outdir_nclar,obs_scheme,".pdf")
  p <- ggplot(mapping=aes(x=time,y=value,colour=iteration),data=d) +
    geom_path(aes(group=iteration)) + geom_hline(aes(yintercept=trueval)) +
    facet_wrap(~component,ncol=1,scales='free_y')+
    scale_colour_gradient(low='green',high='blue')#+
    ylab("")
  p
  
  # write to pdf
  pdf(fn,width=7,height=5)
  show(p)
  dev.off()    
  
