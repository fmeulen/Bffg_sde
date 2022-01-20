using BenchMarkTools

struct MyS 
    a
    b
end




function testje(Ps::Vector{MyS})
    for i in eachindex(Ps)
        @set! Ps[i].a =0
    end
    println(Ps)
end

function testje2(Ps::Vector{MyS})
    for i in eachindex(Ps)
        ui = @set Ps[i].a =0
        Ps[i] = ui[i]
    end
  #  println(Ps)
end


# this is the right one
function testje2b(Ps::Vector{MyS})
    for i in eachindex(Ps)
        ui = Ps[i]
        @set! ui.a = 0
        Ps[i] = ui
    end
    #println(Ps)
end



function testje4(Ps::Vector{MyS})
    for i in eachindex(Ps)
        Ps[i] = @set Ps[i].a =0
    end
    #println(Ps)
end


Ps = [MyS(1,[2,3]), MyS(2.0, 50.0), MyS(-10,10)]
testje(Ps)
Ps


Ps = [MyS(1,[2,3]), MyS(2.0, 50.0), MyS(-10,10)]
@btime testje2(Ps)
Ps

Ps = [MyS(1,[2,3]), MyS(2.0, 50.0), MyS(-10,10)]
@btime testje2b(Ps)  # best
Ps





struct MySvector
    v::Vector{MyS}
end


function testje3(Qs::MySvector)
    for i in eachindex(Qs.v)
         Qs.v[i].a =0
    end
    println(Ps)
end

Qs = MySvector([MyS(1,[2,3]), MyS(2.0, 50.0), MyS(-10,10)])
testje3(Qs)





P1 = P[1]
a, b =P1.a, P1.b
a += 1000
# so I can do calculations with a, but it won't affect P

P[1] = MyS(a,b) # now the new value of a is written in the array
P

c, d = P[2].a, P[2].b
c+= 1000
P = @set P[2].a = c # another way to set the a field of P[2]

function changee(M::MyS)
    a, b = M.a, M.b
    a *=1000.0
    M = @set M.a = a # in fact a new object is made
    M
end


function changeall!(P::Vector{MyS})
    for i in eachindex(P)
        P[i] = changee(P[i])
    end
    return 3, 4
end


u,v = changeall!(P)
P


struct MS
    a
    b
    c
end




function testje(P::MS)
    Pcopy = deepcopy(P)
    a, b, c = P.a, P.b, P.c
    a = 100
    b .= [10, 10]
    c = [100, 100]
    println(Pcopy.a==P.a)
    println(Pcopy.b==P.b)
    println(Pcopy.c==P.c)
    println(Pcopy==P)
    println(Pcopy)
end

P1 = MS(1,[2,3], [4, 5, 6])
testje(P1)


P1.a =100
b = P1.b 
b .= [100 400]


struct M
    a
    b
    c
end

x = [M(1,2,3), M(4,5,6), M(7,8,9)]

ind = [:a, :b]
vals =[10, 300]

# desired output
[M(10,300,3), M(10,300,6), M(10,300,9)]

using Setfield
for i in eachindex(x)
    xi = x[i]
    @set! xi.a=10
    @set! xi.b=300
    x[i] = xi
end
x

x = [M(1,2,3), M(4,5,6), M(7,8,9)]
mytup = (:a => 10, :b => 300)
map!(x, x) do xi
    for (prop, val) ∈ mytup
        l = Setfield.PropertyLens{prop}()
        xi = Setfield.set(xi, l, val)
    end
    xi
end
x

x = [M(1,2,3), M(4,5,6), M(7,8,9)]
y = map(x) do xi
    for (prop, val) ∈ mytup
        l = Setfield.PropertyLens{prop}()
        xi = Setfield.set(xi, l, val)
    end
    xi
end
y

y


x = [𝒫s[1].ℙ, 𝒫s[2].ℙ]
mytup = (:a => 10, :b => 300)
y = map(x) do xi
    for (prop, val) ∈ mytup
        #l = Setfield.PropertyLens{prop}()
        l = Setfield.PropertyLens{:ℙ}();
        (Setfield.PropertyLens{prop}())
        xi = Setfield.set(xi, l, val)
    end
    xi
end
y









using ConstructionBase
struct Q
    z
end

x = [Q(M(1,2,3)), Q(M(4,5,6)), Q(M(7,8,9))]
tup = (a=10,b=300)
for i in eachindex(x)
    x[i] = Q(setproperties(x[i].z, tup))
end
x

tup = (C=1000.0)
for i ∈ eachindex(𝒫s)
    a = 𝒫s[i].ℙ
    xi = ConstructionBase.setproperties(a, tup)
    𝒫s[i].ℙ = xi
end



# construction of named Tuple 
keys = (:a, :b, :c); values = (1, 2, 3);
(; zip(keys, values)...)

(; zip(pars.names, SA[1 2 3])...)