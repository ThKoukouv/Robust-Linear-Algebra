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


n, k = 100, 1
conds_14, conds_15, conds_16, conds_17, conds_18 = [], [], [], [], []
errors_nom_14, errors_minres_14, errors_gmres_14, errors_rob21_14, errors_rob22_14 = [], [], [], [], []
errors_nom_15, errors_minres_15, errors_gmres_15, errors_rob21_15, errors_rob22_15 = [], [], [], [], []
errors_nom_16, errors_minres_16, errors_gmres_16, errors_rob21_16, errors_rob22_16 = [], [], [], [], []
errors_nom_17, errors_minres_17, errors_gmres_17, errors_rob21_17, errors_rob22_17 = [], [], [], [], []
errors_nom_18, errors_minres_18, errors_gmres_18, errors_rob21_18, errors_rob22_18 = [], [], [], [], []


σ = 1e-10
for i in 1:10000
    A = almost_sing(n,k,σ)

    Ainv_nom = inv(A)
    Ainv_minres = compute_inverse_minres(A)
    Ainv_gmres = compute_inverse_gmres(A)
    Ainv_rob21 = rmi_21(A,ρ)
    Ainv_rob22 = rmi_22(A,ρ)


    if cond(A) >= 1e14
        push!(errors_nom_14,norm(A*Ainv_nom - I))
        push!(errors_minres_14,norm(A*Ainv_minres - I))
        push!(errors_gmres_14,norm(A*Ainv_gmres - I))
        push!(errors_rob21_14,norm(A*Ainv_rob21 - I))
        push!(errors_rob22_14,norm(A*Ainv_rob22 - I))
        push!(conds_14,cond(A))
    end
    if length(conds_14) >= 1000
        break
    end
end


σ = 1e-12
for i in 1:10000
    A = almost_sing(n,k,σ)

    Ainv_nom = inv(A)
    Ainv_minres = compute_inverse_minres(A)
    Ainv_gmres = compute_inverse_gmres(A)
    Ainv_rob21 = rmi_21(A,ρ)
    Ainv_rob22 = rmi_22(A,ρ)

    if cond(A) >= 1e16
        push!(errors_nom_16,norm(A*Ainv_nom - I))
        push!(errors_minres_16,norm(A*Ainv_minres - I))
        push!(errors_gmres_16,norm(A*Ainv_gmres - I))
        push!(errors_rob21_16,norm(A*Ainv_rob21 - I))
        push!(errors_rob22_16,norm(A*Ainv_rob22 - I))
        push!(conds_16,cond(A))
    else
        if length(conds_15) <= 1000
            push!(errors_nom_15,norm(A*Ainv_nom - I))
            push!(errors_minres_15,norm(A*Ainv_minres - I))
            push!(errors_gmres_15,norm(A*Ainv_gmres - I))
            push!(errors_rob21_15,norm(A*Ainv_rob21 - I))
            push!(errors_rob22_15,norm(A*Ainv_rob22 - I))
            push!(conds_15,cond(A))
        end
    end
    if (length(conds_15) >= 1000) && (length(conds_16) >= 1000)
        break
    end
end


σ = 1e-14
for i in 1:10000
    A = almost_sing(n,k,σ)
    b = randn(n)

    Ainv_nom = inv(A)
    Ainv_minres = compute_inverse_minres(A)
    Ainv_gmres = compute_inverse_gmres(A)
    Ainv_rob21 = rmi_21(A,ρ)
    Ainv_rob22 = rmi_22(A,ρ)

    if cond(A) >= 1e18
        push!(errors_nom_18,norm(A*Ainv_nom - I))
        push!(errors_minres_18,norm(A*Ainv_minres - I))
        push!(errors_gmres_18,norm(A*Ainv_gmres - I))
        push!(errors_rob21_18,norm(A*Ainv_rob21 - I))
        push!(errors_rob22_18,norm(A*Ainv_rob22 - I))
        push!(conds_18,cond(A))
    else
        if length(conds_17) <= 1000
            push!(errors_nom_17,norm(A*Ainv_nom - I))
            push!(errors_minres_17,norm(A*Ainv_minres - I))
            push!(errors_gmres_17,norm(A*Ainv_gmres - I))
            push!(errors_rob21_17,norm(A*Ainv_rob21 - I))
            push!(errors_rob22_17,norm(A*Ainv_rob22 - I))
            push!(conds_17,cond(A))
        end
    end
    if (length(conds_17) >= 1000) && (length(conds_18) >= 1000)
        break
    end
end

errors_nominal = [mean(errors_nom_15),mean(errors_nom_16),mean(errors_nom_17),mean(errors_nom_18)]
errors_robust = [mean(errors_rob22_15),mean(errors_rob22_16),mean(errors_rob22_17),mean(errors_rob22_18)]
errors_minres = [mean(errors_minres_15),mean(errors_minres_16),mean(errors_minres_17),mean(errors_minres_18)]
errors_gmres = [mean(errors_gmres_15),mean(errors_gmres_16),mean(errors_gmres_17),mean(errors_gmres_18)]

cond_numbers = [1e15,1e16,1e17,1e18]
errors = hcat(errors_nominal,errors_robust,errors_minres,errors_gmres)

writedlm("Cond_Numbers_Inverse.csv", log10.(cond_numbers), ',')
writedlm("Errors_Inverse.csv", errors, ',')


conds = [1e15,1e16,1e17,1e18]
x = log10.(conds)
y = hcat(errors_ci, errors_rmi, errors_minres, errors_gmres)
plot(x, y, label = ["Classical" "MINRES" "GMRES" "Robust"], legend=:topleft)
xlabel!("Matrix condition number")
ylabel!("Norm error")
savefig("RMI_CondNumber_Behavior.png")
