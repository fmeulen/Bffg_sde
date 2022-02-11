# NOTE: Adapted from the implementation found in Julia package
# `DifferentialEquations.jl`

#struct Vern7 <: ODESolverType end

using Parameters


@with_kw struct Vern7Tableau
    c₂::Float64 = 0.005
    c₃::Float64 = 0.10888888888888888
    c₄::Float64 = 0.16333333333333333
    c₅::Float64 = 0.4555
    c₆::Float64 = 0.6095094489978381
    c₇::Float64 = 0.884
    c₈::Float64 = 0.925
    a₂₁::Float64 = 0.005
    a₃₁::Float64 = -1.07679012345679
    a₃₂::Float64 = 1.185679012345679
    a₄₁::Float64 = 0.04083333333333333
    a₄₃::Float64 = 0.1225
    a₅₁::Float64 = 0.6389139236255726
    a₅₃::Float64 = -2.455672638223657
    a₅₄::Float64 = 2.272258714598084
    a₆₁::Float64 = -2.6615773750187572
    a₆₃::Float64 = 10.804513886456137
    a₆₄::Float64 = -8.3539146573962
    a₆₅::Float64 = 0.820487594956657
    a₇₁::Float64 = 6.067741434696772
    a₇₃::Float64 = -24.711273635911088
    a₇₄::Float64 = 20.427517930788895
    a₇₅::Float64 = -1.9061579788166472
    a₇₆::Float64 = 1.006172249242068
    a₈₁::Float64 = 12.054670076253203
    a₈₃::Float64 = -49.75478495046899
    a₈₄::Float64 = 41.142888638604674
    a₈₅::Float64 = -4.461760149974004
    a₈₆::Float64 = 2.042334822239175
    a₈₇::Float64 = -0.09834843665406107
    a₉₁::Float64 = 10.138146522881808
    a₉₃::Float64 = -42.6411360317175
    a₉₄::Float64 = 35.76384003992257
    a₉₅::Float64 = -4.3480228403929075
    a₉₆::Float64 = 2.0098622683770357
    a₉₇::Float64 = 0.3487490460338272
    a₉₈::Float64 = -0.27143900510483127
    a₁₀₁::Float64 = -45.030072034298676
    a₁₀₃::Float64 = 187.3272437654589
    a₁₀₄::Float64 = -154.02882369350186
    a₁₀₅::Float64 = 18.56465306347536
    a₁₀₆::Float64 = -7.141809679295079
    a₁₀₇::Float64 = 1.3088085781613787
    b₁::Float64 = 0.04715561848627222
    b₄::Float64 = 0.25750564298434153
    b₅::Float64 = 0.26216653977412624
    b₆::Float64 = 0.15216092656738558
    b₇::Float64 = 0.4939969170032485
    b₈::Float64 = -0.29430311714032503
    b₉::Float64 = 0.08131747232495111
    b̃₁::Float64 = 0.002547011879931045
    b̃₄::Float64 = -0.00965839487279575
    b̃₅::Float64 = 0.04206470975639691
    b̃₆::Float64 = -0.0666822437469301
    b̃₇::Float64 = 0.2650097464621281
    b̃₈::Float64 = -0.29430311714032503
    b̃₉::Float64 = 0.08131747232495111
    b̃₁₀::Float64 = -0.02029518466335628
end

"""
    createTableau(::Vern7)
Tableau of coefficients for the `Vern7` ODE solver
"""
#createTableau() = Vern7Tableau()

tableau = Vern7Tableau()

# function vern7(f, t, y, dt, P, tableau)
#     (@unpack c₂,c₃,c₄,c₅,c₆,c₇,c₈,a₂₁,a₃₁,a₃₂,a₄₁,a₄₃,a₅₁,a₅₃,a₅₄,a₆₁,a₆₃,a₆₄,
#              a₆₅,a₇₁,a₇₃,a₇₄,a₇₅,a₇₆,a₈₁,a₈₃,a₈₄,a₈₅,a₈₆,a₈₇,a₉₁,a₉₃,a₉₄,a₉₅,
#              a₉₆,a₉₇,a₉₈,b₁,b₄,b₅,b₆,b₇,b₈,b₉ = tableau)
#     k1 = f(t, y, P)
#     k2 = f(t + c₂*dt, y + dt*a₂₁*k1, P)
#     k3 = f(t + c₃*dt, y + dt*(a₃₁*k1 + a₃₂*k2), P)
#     k4 = f(t + c₄*dt, y + dt*(a₄₁*k1 +        + a₄₃*k3), P)
#     k5 = f(t + c₅*dt, y + dt*(a₅₁*k1 +        + a₅₃*k3 + a₅₄*k4), P)
#     k6 = f(t + c₆*dt, y + dt*(a₆₁*k1 +        + a₆₃*k3 + a₆₄*k4 + a₆₅*k5), P)
#     k7 = f(t + c₇*dt, y + dt*(a₇₁*k1 +        + a₇₃*k3 + a₇₄*k4 + a₇₅*k5
#                               + a₇₆*k6), P)
#     k8 = f(t + c₈*dt, y + dt*(a₈₁*k1 +        + a₈₃*k3 + a₈₄*k4 + a₈₅*k5
#                               + a₈₆*k6 + a₈₇*k7), P)
#     k9 = f(t + dt, y + dt*(a₉₁*k1 +        + a₉₃*k3 + a₉₄*k4 + a₉₅*k5
#                            + a₉₆*k6 + a₉₇*k7 + a₉₈*k8), P)
#     y + dt*(b₁*k1 + b₄*k4 + b₅*k5 + b₆*k6 + b₇*k7 + b₈*k8 + b₉*k9)
# end




function vern7(f, t, y::T, dt, P, tableau) where {T}
    (@unpack c₂,c₃,c₄,c₅,c₆,c₇,c₈,a₂₁,a₃₁,a₃₂,a₄₁,a₄₃,a₅₁,a₅₃,a₅₄,a₆₁,a₆₃,a₆₄,
             a₆₅,a₇₁,a₇₃,a₇₄,a₇₅,a₇₆,a₈₁,a₈₃,a₈₄,a₈₅,a₈₆,a₈₇,a₉₁,a₉₃,a₉₄,a₉₅,
             a₉₆,a₉₇,a₉₈,b₁,b₄,b₅,b₆,b₇,b₈,b₉ = tableau)
    k1::T = f(t, y, P)
    k2::T = f(t + c₂*dt, y + dt*a₂₁*k1, P)
    k3::T = f(t + c₃*dt, y + dt*(a₃₁*k1 + a₃₂*k2), P)
    k4::T = f(t + c₄*dt, y + dt*(a₄₁*k1          + a₄₃*k3), P)
    k5::T = f(t + c₅*dt, y + dt*(a₅₁*k1          + a₅₃*k3 + a₅₄*k4), P)
    k6::T = f(t + c₆*dt, y + dt*(a₆₁*k1          + a₆₃*k3 + a₆₄*k4 + a₆₅*k5), P)
    k7::T = f(t + c₇*dt, y + dt*(a₇₁*k1          + a₇₃*k3 + a₇₄*k4 + a₇₅*k5
                              + a₇₆*k6), P)
    k8::T = f(t + c₈*dt, y + dt*(a₈₁*k1          + a₈₃*k3 + a₈₄*k4 + a₈₅*k5
                              + a₈₆*k6 + a₈₇*k7), P)
    k9::T = f(t + dt, y + dt*(a₉₁*k1          + a₉₃*k3 + a₉₄*k4 + a₉₅*k5
                           + a₉₆*k6 + a₉₇*k7 + a₉₈*k8), P)
    (y + dt*(b₁*k1 + b₄*k4 + b₅*k5 + b₆*k6 + b₇*k7 + b₈*k8 + b₉*k9))::T
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
