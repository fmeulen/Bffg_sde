say(what) = run(`osascript -e "say \"$(what)\""`, wait=false)

lastval(X::SamplePath) = X.yy[end]

lastval(ℐ::PathInnovation) = lastval(ℐ.X)

function mergepaths(ℐs)
    tt = map(x->x.X.tt, ℐs)
    yy = map(x->x.X.yy, ℐs)
    SamplePath(vcat(tt...),vcat(yy...))
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



τ(S,T) = (x) ->  S + (x-S) * (2-(x-S)/(T-S))

timegrid(S, T; M) = τ(S,T).(collect(range(S, T, length=M)))

set_timegrids(obs, M) =  [timegrid(obs[i-1].t, obs[i].t, M=M) for i ∈ 2:length(obs)]