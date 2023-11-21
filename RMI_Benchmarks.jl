using LinearAlgebra, JuMP, Gurobi, Random, CSV, IterativeSolvers
using StatsBase, Statistics, DataFrames, Distributions, Distributed
addprocs(8)

function rls_21(A,b,ρ)
    n = size(A)[1]
    m1 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m1, "OutputFlag", 0)
    @variable(m1, x[1:n])
    @variable(m1, t)
    @variable(m1, u[1:n])
    @constraint(m1, [t; A*x .- b] in SecondOrderCone())
    @constraint(m1, x .<= u)
    @constraint(m1, x .>= -u)
    @objective(m1, Min, t + ρ*ones(n)'*u)
    optimize!(m1)
    res = JuMP.value.(x)
    return res
end

function rls_22(A,b,ρ)
    n = size(A)[1]
    m1 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m1, "OutputFlag", 0)
    @variable(m1, x[1:n])
    @variable(m1, t)
    @variable(m1, s)
    @constraint(m1, [t; A*x .- b] in SecondOrderCone())
    @constraint(m1, [s; x] in SecondOrderCone())
    @objective(m1, Min, t + ρ*s)
    optimize!(m1)
    res = JuMP.value.(x)
    return res
end

function rmi_21(A,r)
    n = size(A)[1]
    Ainv = zeros(n,n)
    Threads.@threads for i in 1:n
        e = zeros(n)
        e[i] = 1
        Ainv[:,i] = rls_21(A, e, ρ)
    end
    return Ainv
end

function rmi_22(A,ρ)
    n = size(A)[1]
    Ainv = zeros(n,n)
    Threads.@threads for i in 1:n
        e = zeros(n)
        e[i] = 1
        Ainv[:,i] = rls_22(A, e, ρ)
    end
    return Ainv
end


function compute_inverse_minres(A)
    n = size(A)[1]
    Ainv = zeros(n,n)
    Threads.@threads for i in 1:n
        e = zeros(n)
        e[i] = 1
        Ainv[:,i] = minres(A, e)
    end
    return Ainv
end

function compute_inverse_gmres(A)
    n = size(A)[1]
    Ainv = zeros(n,n)
    Threads.@threads for i in 1:n
        e = zeros(n)
        e[i] = 1
        Ainv[:,i] = gmres(A, e)
    end
    return Ainv
end

#######################   Run experiments    ###########################

n, k, ρ, σ = 100, 1, 0.1, 1e-14

errors_nom_100, errors_minres_100, errors_gmres_100, conds_100 = [], [], [], []
errors_rmi21_100, errors_rmi22_100 = [], []
times_nom_100, times_minres_100, times_gmres_100 = [], [], []
times_rmi21_100, times_rmi22_100 = [], []

for i in 1:100
    A = almost_sing(n,k,σ)

    t1 = time_ns()
    A1 = inv(A)
    t2 = time_ns()
    total_time_nom = (t2-t1)*10^(-9)

    t1 = time_ns()
    A2 = compute_inverse_minres(A)
    t2 = time_ns()
    total_time_minres = (t2-t1)*10^(-9)

    t1 = time_ns()
    A3 = compute_inverse_gmres(A)
    t2 = time_ns()
    total_time_gmres = (t2-t1)*10^(-9)

    t1 = time_ns()
    A4 = rmi_21(A,ρ)
    t2 = time_ns()
    total_time_rmi21 = (t2-t1)*10^(-9)

    t1 = time_ns()
    A5 = rmi_22(A,ρ)
    t2 = time_ns()
    total_time_rmi22 = (t2-t1)*10^(-9)


    push!(errors_nom_100,norm(A*A1-I))
    push!(errors_minres_100,norm(A*A2-I))
    push!(errors_gmres_100,norm(A*A3-I))
    push!(errors_rmi21_100,norm(A*A4-I))
    push!(errors_rmi22_100,norm(A*A5-I))
    push!(conds_100,cond(A))

    push!(times_nom_100,total_time_nom)
    push!(times_minres_100,total_time_minres)
    push!(times_gmres_100,total_time_gmres)
    push!(times_rmi21_100,total_time_rmi21)
    push!(times_rmi22_100,total_time_rmi22)
end


n, k, ρ, σ = 1000, 1, 0.1, 1e-14

errors_nom_1000, errors_minres_1000, errors_gmres_1000, conds_1000 = [], [], [], []
errors_rmi21_1000, errors_rmi22_1000 = [], []
times_nom_1000, times_minres_1000, times_gmres_1000 = [], [], []
times_rmi21_1000, times_rmi22_1000 = [], []

for i in 1:10
    A = almost_sing(n,k,σ)

    t1 = time_ns()
    A1 = inv(A)
    t2 = time_ns()
    total_time_nom = (t2-t1)*10^(-9)

    t1 = time_ns()
    A2 = compute_inverse_minres(A)
    t2 = time_ns()
    total_time_minres = (t2-t1)*10^(-9)

    t1 = time_ns()
    A3 = compute_inverse_gmres(A)
    t2 = time_ns()
    total_time_gmres = (t2-t1)*10^(-9)

    t1 = time_ns()
    A4 = rmi_21(A,ρ)
    t2 = time_ns()
    total_time_rmi21 = (t2-t1)*10^(-9)

    t1 = time_ns()
    A5 = rmi_22(A,ρ)
    t2 = time_ns()
    total_time_rmi22 = (t2-t1)*10^(-9)


    push!(errors_nom_1000,norm(A*A1-I))
    push!(errors_minres_1000,norm(A*A2-I))
    push!(errors_gmres_1000,norm(A*A3-I))
    push!(errors_rmi21_1000,norm(A*A4-I))
    push!(errors_rmi22_1000,norm(A*A5-I))
    push!(conds_1000,cond(A))

    push!(times_nom_1000,total_time_nom)
    push!(times_minres_1000,total_time_minres)
    push!(times_gmres_1000,total_time_gmres)
    push!(times_rmi21_1000,total_time_rmi21)
    push!(times_rmi22_1000,total_time_rmi22)
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
println(mean(times_rmi22_100))
println(std(times_rmi22_100))
println("RLS-21:")
println(mean(times_rmi21_100))
println(std(times_rmi21_100))


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
println(mean(times_rmi22_1000))
println(std(times_rmi22_1000))
println("RLS-21:")
println(mean(times_rmi21_1000))
println(std(times_rmi21_1000))

#############   Glasso comparisons  ###################

using RCall
R"library(glasso)"


errors_nom = []
errors_rmi_001, errors_rmi_01, errors_rmi_03 = [], [], []
errors_rmi_05, errors_rmi_07, errors_rmi_1 = [], [], []
errors_glasso_001, errors_glasso_01, errors_glasso_03 = [], [], []
errors_glasso_05, errors_glasso_07, errors_glasso_1 = [], [], []

for i in 1:10
    X = randn(100,105)
    C = cov(X)
    Cinv_nom = inv(C)
    Cinv_rmi_001 = rmi_22(C,0.01)
    Cinv_rmi_01 = rmi_22(C,0.1)
    Cinv_rmi_03 = rmi_22(C,0.3)
    Cinv_rmi_05 = rmi_22(C,0.5)
    Cinv_rmi_07 = rmi_22(C,0.7)
    Cinv_rmi_1 = rmi_22(C,1)
    @rput C
    R"r=c(0.01,0.1,0.3,0.5,0.7,1.0)"
    R"y = glassopath(C,rho=r)"
    R"z1 <- y$wi[, ,1]"
    R"z2 <- y$wi[, ,2]"
    R"z3 <- y$wi[, ,3]"
    R"z4 <- y$wi[, ,4]"
    R"z5 <- y$wi[, ,5]"
    R"z6 <- y$wi[, ,6]"
    @rget z1
    @rget z2
    @rget z3
    @rget z4
    @rget z5
    @rget z6
    Cinv_glasso_001 = z1
    Cinv_glasso_01 = z2
    Cinv_glasso_03 = z3
    Cinv_glasso_05 = z4
    Cinv_glasso_07 = z5
    Cinv_glasso_1 = z6
    push!(errors_nom, norm(C*Cinv_nom - I))
    push!(errors_rmi_001, norm(C*Cinv_rmi_001 - I))
    push!(errors_glasso_001, norm(C*Cinv_glasso_001 - I))
    push!(errors_rmi_01, norm(C*Cinv_rmi_01 - I))
    push!(errors_glasso_01, norm(C*Cinv_glasso_01 - I))
    push!(errors_rmi_03, norm(C*Cinv_rmi_03 - I))
    push!(errors_glasso_03, norm(C*Cinv_glasso_03 - I))
    push!(errors_rmi_05, norm(C*Cinv_rmi_05 - I))
    push!(errors_glasso_05, norm(C*Cinv_glasso_05 - I))
    push!(errors_rmi_07, norm(C*Cinv_rmi_07 - I))
    push!(errors_glasso_07, norm(C*Cinv_glasso_07 - I))
    push!(errors_rmi_1, norm(C*Cinv_rmi_1 - I))
    push!(errors_glasso_1, norm(C*Cinv_glasso_1 - I))
end


println("Nominal:")
println(norm(C*Cinv_nom - I))
println("Robust, rho=0.01:")
println("RMI:")
println(mean(errors_rmi_001))
println(std(errors_rmi_001))
println("Glasso:")
println(mean(errors_glasso_001))
println(std(errors_glasso_001))
println("Robust, rho=0.1:")
println("RMI:")
println(mean(errors_rmi_01))
println(std(errors_rmi_01))
println("Glasso:")
println(mean(errors_glasso_01))
println(std(errors_glasso_01))
println("Robust, rho=0.3:")
println("RMI:")
println(mean(errors_rmi_03))
println(std(errors_rmi_03))
println("Glasso:")
println(mean(errors_glasso_03))
println(std(errors_glasso_03))
println("Robust, rho=0.5:")
println("RMI:")
println(mean(errors_rmi_05))
println(std(errors_rmi_05))
println("Glasso:")
println(mean(errors_glasso_05))
println(std(errors_glasso_05))
println("Robust, rho=0.7:")
println("RMI:")
println(mean(errors_rmi_07))
println(std(errors_rmi_07))
println("Glasso:")
println(mean(errors_glasso_07))
println(std(errors_glasso_07))
println("Robust, rho=1:")
println("RMI:")
println(mean(errors_rmi_1))
println(std(errors_rmi_1))
println("Glasso:")
println(mean(errors_glasso_1))
println(std(errors_glasso_1))


errors_nom = []
errors_rmi_001, errors_rmi_01, errors_rmi_03 = [], [], []
errors_rmi_05, errors_rmi_07, errors_rmi_1 = [], [], []
errors_glasso_001, errors_glasso_01, errors_glasso_03 = [], [], []
errors_glasso_05, errors_glasso_07, errors_glasso_1 = [], [], []

for i in 1:10
    X = randn(500,550)
    C = cov(X)
    Cinv_nom = inv(C)
    Cinv_rmi_001 = rmi_22(C,0.01)
    Cinv_rmi_01 = rmi_22(C,0.1)
    Cinv_rmi_03 = rmi_22(C,0.3)
    Cinv_rmi_05 = rmi_22(C,0.5)
    Cinv_rmi_07 = rmi_22(C,0.7)
    Cinv_rmi_1 = rmi_22(C,1)
    @rput C
    R"r=c(0.01,0.1,0.3,0.5,0.7,1.0)"
    R"y = glassopath(C,rho=r)"
    R"z1 <- y$wi[, ,1]"
    R"z2 <- y$wi[, ,2]"
    R"z3 <- y$wi[, ,3]"
    R"z4 <- y$wi[, ,4]"
    R"z5 <- y$wi[, ,5]"
    R"z6 <- y$wi[, ,6]"
    @rget z1
    @rget z2
    @rget z3
    @rget z4
    @rget z5
    @rget z6
    Cinv_glasso_001 = z1
    Cinv_glasso_01 = z2
    Cinv_glasso_03 = z3
    Cinv_glasso_05 = z4
    Cinv_glasso_07 = z5
    Cinv_glasso_1 = z6
    push!(errors_nom, norm(C*Cinv_nom - I))
    push!(errors_rmi_001, norm(C*Cinv_rmi_001 - I))
    push!(errors_glasso_001, norm(C*Cinv_glasso_001 - I))
    push!(errors_rmi_01, norm(C*Cinv_rmi_01 - I))
    push!(errors_glasso_01, norm(C*Cinv_glasso_01 - I))
    push!(errors_rmi_03, norm(C*Cinv_rmi_03 - I))
    push!(errors_glasso_03, norm(C*Cinv_glasso_03 - I))
    push!(errors_rmi_05, norm(C*Cinv_rmi_05 - I))
    push!(errors_glasso_05, norm(C*Cinv_glasso_05 - I))
    push!(errors_rmi_07, norm(C*Cinv_rmi_07 - I))
    push!(errors_glasso_07, norm(C*Cinv_glasso_07 - I))
    push!(errors_rmi_1, norm(C*Cinv_rmi_1 - I))
    push!(errors_glasso_1, norm(C*Cinv_glasso_1 - I))
end


println("Nominal:")
println(norm(C*Cinv_nom - I))
println("Robust, rho=0.01:")
println("RMI:")
println(mean(errors_rmi_001))
println(std(errors_rmi_001))
println("Glasso:")
println(mean(errors_glasso_001))
println(std(errors_glasso_001))
println("Robust, rho=0.1:")
println("RMI:")
println(mean(errors_rmi_01))
println(std(errors_rmi_01))
println("Glasso:")
println(mean(errors_glasso_01))
println(std(errors_glasso_01))
println("Robust, rho=0.3:")
println("RMI:")
println(mean(errors_rmi_03))
println(std(errors_rmi_03))
println("Glasso:")
println(mean(errors_glasso_03))
println(std(errors_glasso_03))
println("Robust, rho=0.5:")
println("RMI:")
println(mean(errors_rmi_05))
println(std(errors_rmi_05))
println("Glasso:")
println(mean(errors_glasso_05))
println(std(errors_glasso_05))
println("Robust, rho=0.7:")
println("RMI:")
println(mean(errors_rmi_07))
println(std(errors_rmi_07))
println("Glasso:")
println(mean(errors_glasso_07))
println(std(errors_glasso_07))
println("Robust, rho=1:")
println("RMI:")
println(mean(errors_rmi_1))
println(std(errors_rmi_1))
println("Glasso:")
println(mean(errors_glasso_1))
println(std(errors_glasso_1))





errors_nom = []
errors_rmi_001, errors_rmi_01, errors_rmi_03 = [], [], []
errors_rmi_05, errors_rmi_07, errors_rmi_1 = [], [], []
errors_glasso_001, errors_glasso_01, errors_glasso_03 = [], [], []
errors_glasso_05, errors_glasso_07, errors_glasso_1 = [], [], []

for i in 1:10
    X = randn(1500,1600)
    C = cov(X)
    Cinv_nom = inv(C)
    Cinv_rmi_001 = rmi_22(C,0.01)
    Cinv_rmi_01 = rmi_22(C,0.1)
    Cinv_rmi_03 = rmi_22(C,0.3)
    Cinv_rmi_05 = rmi_22(C,0.5)
    Cinv_rmi_07 = rmi_22(C,0.7)
    Cinv_rmi_1 = rmi_22(C,1)
    @rput C
    R"r=c(0.01,0.1,0.3,0.5,0.7,1.0)"
    R"y = glassopath(C,rho=r)"
    R"z1 <- y$wi[, ,1]"
    R"z2 <- y$wi[, ,2]"
    R"z3 <- y$wi[, ,3]"
    R"z4 <- y$wi[, ,4]"
    R"z5 <- y$wi[, ,5]"
    R"z6 <- y$wi[, ,6]"
    @rget z1
    @rget z2
    @rget z3
    @rget z4
    @rget z5
    @rget z6
    Cinv_glasso_001 = z1
    Cinv_glasso_01 = z2
    Cinv_glasso_03 = z3
    Cinv_glasso_05 = z4
    Cinv_glasso_07 = z5
    Cinv_glasso_1 = z6
    push!(errors_nom, norm(C*Cinv_nom - I))
    push!(errors_rmi_001, norm(C*Cinv_rmi_001 - I))
    push!(errors_glasso_001, norm(C*Cinv_glasso_001 - I))
    push!(errors_rmi_01, norm(C*Cinv_rmi_01 - I))
    push!(errors_glasso_01, norm(C*Cinv_glasso_01 - I))
    push!(errors_rmi_03, norm(C*Cinv_rmi_03 - I))
    push!(errors_glasso_03, norm(C*Cinv_glasso_03 - I))
    push!(errors_rmi_05, norm(C*Cinv_rmi_05 - I))
    push!(errors_glasso_05, norm(C*Cinv_glasso_05 - I))
    push!(errors_rmi_07, norm(C*Cinv_rmi_07 - I))
    push!(errors_glasso_07, norm(C*Cinv_glasso_07 - I))
    push!(errors_rmi_1, norm(C*Cinv_rmi_1 - I))
    push!(errors_glasso_1, norm(C*Cinv_glasso_1 - I))
end


println("Nominal:")
println(norm(C*Cinv_nom - I))
println("Robust, rho=0.01:")
println("RMI:")
println(mean(errors_rmi_001))
println(std(errors_rmi_001))
println("Glasso:")
println(mean(errors_glasso_001))
println(std(errors_glasso_001))
println("Robust, rho=0.1:")
println("RMI:")
println(mean(errors_rmi_01))
println(std(errors_rmi_01))
println("Glasso:")
println(mean(errors_glasso_01))
println(std(errors_glasso_01))
println("Robust, rho=0.3:")
println("RMI:")
println(mean(errors_rmi_03))
println(std(errors_rmi_03))
println("Glasso:")
println(mean(errors_glasso_03))
println(std(errors_glasso_03))
println("Robust, rho=0.5:")
println("RMI:")
println(mean(errors_rmi_05))
println(std(errors_rmi_05))
println("Glasso:")
println(mean(errors_glasso_05))
println(std(errors_glasso_05))
println("Robust, rho=0.7:")
println("RMI:")
println(mean(errors_rmi_07))
println(std(errors_rmi_07))
println("Glasso:")
println(mean(errors_glasso_07))
println(std(errors_glasso_07))
println("Robust, rho=1:")
println("RMI:")
println(mean(errors_rmi_1))
println(std(errors_rmi_1))
println("Glasso:")
println(mean(errors_glasso_1))
println(std(errors_glasso_1))
