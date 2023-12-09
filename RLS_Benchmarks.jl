using LinearAlgebra, JuMP, SCS, Statistics, StatsBase, Distributions
using Mosek, MosekTools, IterativeSolvers, Random, Distributed


#######################   Main functions   #############################

function create_data(n,k)
    B = zeros(k,n)
    A = rand(n-k,n)
    m = convert(Int64, round(n/2))
    indices = sample(1:m,k,replace=false)
    l = length(indices)
    eps = 1e-2
    for i in 1:l
        B[i,:] = A[indices[i],:] .+ eps # .+ sum(A[indices[i],j] for j in 1:n)
    end
    C = vcat(A,B)
end

function almost_sing(n,k,σ)
    d = Normal(0.0,σ)
    B = rand(n-k,n)
    C = B
    for i in 1:k
        eps = rand(d,n)
        b = sum(B[i,:] for i in 1:n-k) .+ eps
        C = vcat(C,b')
    end
    return C
end

function rls_11(A,b,ρ)
    n = size(A)[1]
    m1 = Model(SCS.Optimizer)
    io = open("/dev/null", "w")  # On Unix-like systems, redirect to null device
    redirect_stdout(io)
    @variable(m1, x[1:n])
    @variable(m1, w[1:n])
    @variable(m1, u[1:n])
    @constraint(m1, A*x .- b .<= w)
    @constraint(m1, A*x .- b .>= -w)
    @constraint(m1, x .<= u)
    @constraint(m1, x .>= -u)
    @objective(m1, Min, ones(n)'*w + ρ*ones(n)'*u)
    optimize!(m1)
    close(io)
    res = JuMP.value.(x)
    return res
end


function rls_21(A,b,ρ)
    n = size(A)[1]
    m1 = Model(SCS.Optimizer)
    io = open("/dev/null", "w")  # On Unix-like systems, redirect to null device
    redirect_stdout(io)
    @variable(m1, x[1:n])
    @variable(m1, t)
    @variable(m1, u[1:n])
    @constraint(m1, [t; A*x .- b] in SecondOrderCone())
    @constraint(m1, x .<= u)
    @constraint(m1, x .>= -u)
    @objective(m1, Min, t + ρ*ones(n)'*u)
    optimize!(m1)
    close(io)
    res = JuMP.value.(x)
    return res
end

function rls_22(A,b,ρ)
    n = size(A)[1]
    m1 = Model(SCS.Optimizer)
    io = open("/dev/null", "w")  # On Unix-like systems, redirect to null device
    redirect_stdout(io)
    @variable(m1, x[1:n])
    @variable(m1, t)
    @variable(m1, s)
    @constraint(m1, [t; A*x .- b] in SecondOrderCone())
    @constraint(m1, [s; x] in SecondOrderCone())
    @objective(m1, Min, t + ρ*s)
    optimize!(m1)
    close(io)
    res = JuMP.value.(x)
    return res
end

#####################    Run Experiment    #############################

n, k, ρ, σ = 100, 1, 0.1, 1e-13

errors_nom_100, errors_minres_100, errors_gmres_100, conds_100 = [], [], [], []
errors_rob21_100, errors_rob22_100, errors_rob11_100 = [], [], []
times_nom_100, times_minres_100, times_gmres_100 = [], [], []
times_rob21_100, times_rob22_100, times_rob11_100 = [], [], []

for i in 1:100
    A = almost_sing(n,k,σ)
    b = randn(n)

    t1 = time_ns()
    x1 = A \ b
    t2 = time_ns()
    total_time_nom = (t2-t1)*10^(-9)

    t1 = time_ns()
    x2 = minres(A, b)
    t2 = time_ns()
    total_time_minres = (t2-t1)*10^(-9)

    t1 = time_ns()
    x3 = gmres(A, b)
    t2 = time_ns()
    total_time_gmres = (t2-t1)*10^(-9)

    t1 = time_ns()
    x4 = rls_21(A,b,ρ)
    t2 = time_ns()
    total_time_rls21 = (t2-t1)*10^(-9)

    t1 = time_ns()
    x5 = rls_22(A,b,ρ)
    t2 = time_ns()
    total_time_rls22 = (t2-t1)*10^(-9)

    t1 = time_ns()
    x6 = rls_11(A,b,ρ)
    t2 = time_ns()
    total_time_rls11 = (t2-t1)*10^(-9)

    push!(errors_nom_100,norm(A*x1.-b))
    push!(errors_minres_100,norm(A*x2.-b))
    push!(errors_gmres_100,norm(A*x3.-b))
    push!(errors_rob21_100,norm(A*x4.-b))
    push!(errors_rob22_100,norm(A*x5.-b))
    push!(errors_rob11_100,norm(A*x6.-b))
    push!(conds_100,cond(A))

    push!(times_nom_100,total_time_nom)
    push!(times_minres_100,total_time_minres)
    push!(times_gmres_100,total_time_gmres)
    push!(times_rob21_100,total_time_rls21)
    push!(times_rob22_100,total_time_rls22)
    push!(times_rob11_100,total_time_rls11)
end


n, k, ρ, σ = 1000, 1, 0.1, 1e-13

errors_nom_1000, errors_minres_1000, errors_gmres_1000, conds_1000 = [], [], [], []
errors_rob21_1000, errors_rob22_1000, errors_rob11_1000 = [], [], []
times_nom_1000, times_minres_1000, times_gmres_1000 = [], [], []
times_rob21_1000, times_rob22_1000, times_rob11_1000 = [], [], []

for i in 1:100
    A = almost_sing(n,k,σ)
    b = randn(n)

    t1 = time_ns()
    x1 = A \ b
    t2 = time_ns()
    total_time_nom = (t2-t1)*10^(-9)

    t1 = time_ns()
    x2 = minres(A, b)
    t2 = time_ns()
    total_time_minres = (t2-t1)*10^(-9)

    t1 = time_ns()
    x3 = gmres(A, b)
    t2 = time_ns()
    total_time_gmres = (t2-t1)*10^(-9)

    t1 = time_ns()
    x4 = rls_21(A,b,ρ)
    t2 = time_ns()
    total_time_rls21 = (t2-t1)*10^(-9)

    t1 = time_ns()
    x5 = rls_22(A,b,ρ)
    t2 = time_ns()
    total_time_rls22 = (t2-t1)*10^(-9)

    t1 = time_ns()
    x6 = rls_11(A,b,ρ)
    t2 = time_ns()
    total_time_rls11 = (t2-t1)*10^(-9)

    push!(errors_nom_1000,norm(A*x1.-b))
    push!(errors_minres_1000,norm(A*x2.-b))
    push!(errors_gmres_1000,norm(A*x3.-b))
    push!(errors_rob21_1000,norm(A*x4.-b))
    push!(errors_rob22_1000,norm(A*x5.-b))
    push!(errors_rob11_1000,norm(A*x6.-b))
    push!(conds_1000,cond(A))

    push!(times_nom_1000,total_time_nom)
    push!(times_minres_1000,total_time_minres)
    push!(times_gmres_1000,total_time_gmres)
    push!(times_rob21_1000,total_time_rls21)
    push!(times_rob22_1000,total_time_rls22)
    push!(times_rob11_1000,total_time_rls11)


    println("-------------------------")
    println(i)
end

n, k, ρ, σ = 10000, 1, 0.1, 1e-14

errors_nom_10000, errors_minres_10000, errors_gmres_10000, conds_10000 = [], [], [], []
errors_rob21_10000, errors_rob22_10000 = [], []
times_nom_10000, times_minres_10000, times_gmres_10000 = [], [], []
times_rob21_10000, times_rob22_10000 = [], []

for i in 1:10
    A = almost_sing(n,k,σ)
    b = randn(n)

    t1 = time_ns()
    x1 = A \ b
    t2 = time_ns()
    total_time_nom = (t2-t1)*10^(-9)

    t1 = time_ns()
    x2 = minres(A, b)
    t2 = time_ns()
    total_time_minres = (t2-t1)*10^(-9)

    t1 = time_ns()
    x3 = gmres(A, b)
    t2 = time_ns()
    total_time_gmres = (t2-t1)*10^(-9)

    t1 = time_ns()
    x4 = rls_21(A,b,ρ)
    t2 = time_ns()
    total_time_rls21 = (t2-t1)*10^(-9)

    t1 = time_ns()
    x5 = rls_22(A,b,ρ)
    t2 = time_ns()
    total_time_rls22 = (t2-t1)*10^(-9)

    t1 = time_ns()
    x6 = rls_11(A,b,ρ)
    t2 = time_ns()
    total_time_rls11 = (t2-t1)*10^(-9)


    push!(errors_nom_10000,norm(A*x1.-b))
    push!(errors_minres_10000,norm(A*x2.-b))
    push!(errors_gmres_10000,norm(A*x3.-b))
    push!(errors_rob21_10000,norm(A*x4.-b))
    push!(errors_rob22_10000,norm(A*x5.-b))
    push!(conds_10000,cond(A))

    push!(times_nom_10000,total_time_nom)
    push!(times_minres_10000,total_time_minres)
    push!(times_gmres_10000,total_time_gmres)
    push!(times_rob21_10000,total_time_rls21)
    push!(times_rob22_10000,total_time_rls22)
    push!(times_rob11_10000,total_time_rls11)


    println("-------------------------")
    println(i)
end

println("n=100")
println("Nominal:")
println(mean(times_nom_100))
println(std(times_nom_100))
println("MINRES:")
println(mean(times_minres_100))
println(std(times_minres_100))
println("GMRES:")
println(mean(times_gmres_100))
println(std(times_gmres_100))
println("RLS-22:")
println(mean(times_rob22_100))
println(std(times_rob22_100))
println("RLS-21:")
println(mean(times_rob21_100))
println(std(times_rob21_100))
println("RLS-11:")
println(mean(times_rob11_100))
println(std(times_rob11_100))


println("n=1000")
println("Nominal:")
println(mean(times_nom_1000))
println(std(times_nom_1000))
println("MINRES:")
println(mean(times_minres_1000))
println(std(times_minres_1000))
println("GMRES:")
println(mean(times_gmres_1000))
println(std(times_gmres_1000))
println("RLS-22:")
println(mean(times_rob22_1000))
println(std(times_rob22_1000))
println("RLS-21:")
println(mean(times_rob21_1000))
println(std(times_rob21_1000))
println("RLS-11:")
println(mean(times_rob11_1000))
println(std(times_rob11_1000))

println("n=10000")
println("Nominal:")
println(mean(times_nom_10000))
println(std(times_nom_10000))
println("MINRES:")
println(mean(times_minres_10000))
println(std(times_minres_10000))
println("GMRES:")
println(mean(times_gmres_10000))
println(std(times_gmres_10000))
println("RLS-22:")
println(mean(times_rob22_10000))
println(std(times_rob22_10000))
println("RLS-21:")
println(mean(times_rob21_10000))
println(std(times_rob21_10000))
println("RLS-11:")
println(mean(times_rob11_1000))
println(std(times_rob11_1000))
