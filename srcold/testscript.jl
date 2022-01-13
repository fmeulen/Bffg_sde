struct MyS 
    a
    b
end

P = [MyS(1,[2,3]), MyS(2.0, 50.0), MyS(-10,10)]

 Q = [MyS(1000,1), MyS("lala",0), MyS(1,0.0)]


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