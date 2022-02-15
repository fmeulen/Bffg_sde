module Bffg_sde # GuidingStochDiffEq

using Bridge
using StaticArrays
using Random 
using LinearAlgebra
using Distributions
using Bridge.Models
using DifferentialEquations
using Setfield
using ConstructionBase
using Interpolations
using IterTools

const sk=0

include("tableaus_ode_solvers.jl")

include("types.jl")
export ParMove, Innovations, Observation, Htransform, Message, BackwardFilter, State

include("forwardguiding.jl")
export forwardguide

include("backwardfiltering.jl")

include("utilities.jl")
export timegrids, say, printinfo

include("parameter_path_updates.jl")
export setpar, getpar, propose, logpriordiff, pcnupdate!, parupdate!, exploremoveσfixed!


end # module