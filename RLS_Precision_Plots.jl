using LinearAlgebra, JuMP, Gurobi, Statistics, StatsBase, Distributions
using Mosek, MosekTools, IterativeSolvers, Random, Distributed


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
    m1 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m1, "OutputFlag", 0)
    @variable(m1, x[1:n])
    @variable(m1, w[1:n])
    @variable(m1, u[1:n])
    @constraint(m1, A*x .- b .<= w)
    @constraint(m1, A*x .- b .>= -w)
    @constraint(m1, x .<= u)
    @constraint(m1, x .>= -u)
    @objective(m1, Min, ones(n)'*w + ρ*ones(n)'*u)
    optimize!(m1)
    res = JuMP.value.(x)
    return res
end


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


########################   Double Precision   ###########################

n, k, ρ = 100, 1, 0.1
σ = 1e-13

errors_nom_16, errors_minres_16, errors_gmres_16, errors_rob_16, conds_16 = [], [], [], [], []
errors_nom_17, errors_minres_17, errors_gmres_17, errors_rob_17, conds_17 = [], [], [], [], []

for i in 1:10000
    A = almost_sing(n,k,σ)
    b = randn(n)

    x1 = A \ b
    x2 = minres(A, b)
    x3 = gmres(A, b)
    x4 = rls_11(A,b,ρ)
    x5 = rls_21(A,b,ρ)
    x6 = rls_22(A,b,ρ)

    if cond(A) >= 1e17
        push!(errors_nom_17,norm(A*x1.-b))
        push!(errors_minres_17,norm(A*x2.-b))
        push!(errors_gmres_17,norm(A*x3.-b))
        push!(errors_rob_17,norm(A*x6.-b))
        push!(conds_17,cond(A))
    else
        push!(errors_nom_16,norm(A*x1.-b))
        push!(errors_minres_16,norm(A*x2.-b))
        push!(errors_gmres_16,norm(A*x3.-b))
        push!(errors_rob_16,norm(A*x6.-b))
        push!(conds_16,cond(A))
    end
end

conds_16
mean(errors_nom_16)
mean(errors_minres_16)
mean(errors_gmres_16)
mean(errors_rob_16)

conds_17
mean(errors_nom_17)
mean(errors_minres_17)
mean(errors_gmres_17)
mean(errors_rob_17)



σ = 1e-14

errors_nom_18, errors_minres_18, errors_gmres_18, errors_rob_18, conds_18 = [], [], [], [], []

for i in 1:100
    A = almost_sing(n,k,σ)
    b = randn(n)

    x1 = A \ b
    x2 = minres(A, b)
    x3 = gmres(A, b)
    x4 = rls_11(A,b,ρ)
    x5 = rls_21(A,b,ρ)
    x6 = rls_22(A,b,ρ)

    if cond(A) >= 1e18
        push!(errors_nom_18,norm(A*x1.-b))
        push!(errors_minres_18,norm(A*x2.-b))
        push!(errors_gmres_18,norm(A*x3.-b))
        push!(errors_rob_18,norm(A*x6.-b))
        push!(conds_18,cond(A))
    end
end


mean(errors_nom_18)
mean(errors_minres_18)
mean(errors_gmres_18)
mean(errors_rob_18)


# n = 100
n, k, σ, ρ = 100, 2, 1e-13, 0.01

errors_nom, errors_minres, errors_gmres = [], [], []
errors_rob_11, errors_rob_21, errors_rob_22, conds = [], [], [], [], []

for i in 1:100
    A = almost_sing(n,k,σ)
    b = randn(n)

    push!(conds,cond(A))

    x1 = A \ b
    x2 = minres(A, b)
    x3 = gmres(A, b)
    x4 = rls_11(A,b,ρ)
    x5 = rls_21(A,b,ρ)
    x6 = rls_22(A,b,ρ)


    push!(errors_nom,norm(A*x1.-b))
    push!(errors_minres,norm(A*x2.-b))
    push!(errors_gmres,norm(A*x3.-b))
    push!(errors_rob_11,norm(A*x4.-b))
    push!(errors_rob_21,norm(A*x5.-b))
    push!(errors_rob_22,norm(A*x6.-b))
end

mean(errors_nom)
mean(errors_minres)
mean(errors_gmres)
mean(errors_rob_11)
mean(errors_rob_21)
mean(errors_rob_22)

std(errors_nom)
std(errors_minres)
std(errors_gmres)
std(errors_rob_11)
std(errors_rob_21)
std(errors_rob_22)


# n = 1000
n, k, σ, ρ = 1000, 1, 1e-13, 0.01

errors_nom, errors_minres, errors_gmres = [], [], []
errors_rob_11, errors_rob_21, errors_rob_22, conds = [], [], [], [], []

for i in 1:100
    A = almost_sing(n,k,σ)
    b = randn(n)

    push!(conds,cond(A))

    x1 = A \ b
    x2 = minres(A, b)
    x3 = gmres(A, b)
    x4 = rls_11(A,b,ρ)
    x5 = rls_21(A,b,ρ)
    x6 = rls_22(A,b,ρ)


    push!(errors_nom,norm(A*x1.-b))
    push!(errors_minres,norm(A*x2.-b))
    push!(errors_gmres,norm(A*x3.-b))
    push!(errors_rob_11,norm(A*x4.-b))
    push!(errors_rob_21,norm(A*x5.-b))
    push!(errors_rob_22,norm(A*x6.-b))
end

mean(errors_nom)
mean(errors_minres)
mean(errors_gmres)
mean(errors_rob_11)
mean(errors_rob_21)
mean(errors_rob_22)

std(errors_nom)
std(errors_minres)
std(errors_gmres)
std(errors_rob_11)
std(errors_rob_21)
std(errors_rob_22)



#########################   Single Precision    ###########################

n, k = 100, 1

conds_5, conds_6, conds_7, conds_8, conds_9 = [], [], [], [], []
errors_nom_5, errors_minres_5, errors_gmres_5, errors_rob21_5, errors_rob22_5 = [], [], [], [], []
errors_nom_6, errors_minres_6, errors_gmres_6, errors_rob21_6, errors_rob22_6 = [], [], [], [], []
errors_nom_7, errors_minres_7, errors_gmres_7, errors_rob21_7, errors_rob22_7 = [], [], [], [], []
errors_nom_8, errors_minres_8, errors_gmres_8, errors_rob21_8, errors_rob22_8 = [], [], [], [], []
errors_nom_9, errors_minres_9, errors_gmres_9, errors_rob21_9, errors_rob22_9 = [], [], [], [], []


σ = 1e-2
for i in 1:10000
    A = Float32.(almost_sing(n,k,σ))
    b = Float32.(randn(n))

    x1 = A \ b
    x2 = minres(A, b)
    x3 = gmres(A, b)
    x4 = rls_21(A,b,ρ)
    x5 = rls_22(A,b,ρ)

    if cond(A) <= 1e6
        push!(errors_nom_5,norm(A*x1.-b))
        push!(errors_minres_5,norm(A*x2.-b))
        push!(errors_gmres_5,norm(A*x3.-b))
        push!(errors_rob21_5,norm(A*x4.-b))
        push!(errors_rob22_5,norm(A*x5.-b))
        push!(conds_5,cond(A))
    end
    if length(conds_5) == 1000
        break
    end
end



σ = 1e-3
for i in 1:10000
    A = Float32.(almost_sing(n,k,σ))
    b = Float32.(randn(n))

    x1 = A \ b
    x2 = minres(A, b)
    x3 = gmres(A, b)
    x4 = rls_21(A,b,ρ)
    x5 = rls_22(A,b,ρ)

    if cond(A) >= 1e7
        push!(errors_nom_7,norm(A*x1.-b))
        push!(errors_minres_7,norm(A*x2.-b))
        push!(errors_gmres_7,norm(A*x3.-b))
        push!(errors_rob21_7,norm(A*x4.-b))
        push!(errors_rob22_7,norm(A*x5.-b))
        push!(conds_7,cond(A))
    else
        push!(errors_nom_6,norm(A*x1.-b))
        push!(errors_minres_6,norm(A*x2.-b))
        push!(errors_gmres_6,norm(A*x3.-b))
        push!(errors_rob21_6,norm(A*x4.-b))
        push!(errors_rob22_6,norm(A*x5.-b))
        push!(conds_6,cond(A))
    end
    if (length(conds_6) >= 1000) && (length(conds_7) >= 1000)
        break
    end
end


σ = 1e-5
for i in 1:10000
    A = Float32.(almost_sing(n,k,σ))
    b = Float32.(randn(n))

    x1 = A \ b
    x2 = minres(A, b)
    x3 = gmres(A, b)
    x4 = rls_21(A,b,ρ)
    x5 = rls_22(A,b,ρ)

    if cond(A) >= 1e9
        push!(errors_nom_9,norm(A*x1.-b))
        push!(errors_minres_9,norm(A*x2.-b))
        push!(errors_gmres_9,norm(A*x3.-b))
        push!(errors_rob21_9,norm(A*x4.-b))
        push!(errors_rob22_9,norm(A*x5.-b))
        push!(conds_9,cond(A))
    else
        push!(errors_nom_8,norm(A*x1.-b))
        push!(errors_minres_8,norm(A*x2.-b))
        push!(errors_gmres_8,norm(A*x3.-b))
        push!(errors_rob21_8,norm(A*x4.-b))
        push!(errors_rob22_8,norm(A*x5.-b))
        push!(conds_8,cond(A))
    end
    if (length(conds_8) >= 1000) && (length(conds_9) >= 1000)
        break
    end
end


errors_nominal = [mean(errors_nom_5),mean(errors_nom_6),mean(errors_nom_7),mean(errors_nom_8),mean(errors_nom_9)]
errors_robust = [mean(errors_rob22_5),mean(errors_rob22_6),mean(errors_rob22_7),mean(errors_rob22_8),mean(errors_rob22_9)]

cond_numbers = [1e5,1e6,1e7,1e8,1e9]
errors = hcat(errors_nominal,errors_robust)

plot(log10.(cond_numbers), errors, title = "Single Precision Results",  label = ["Nominal" "Robust"], legend=:topleft)
xlabel!("log10(cond(A))")
ylabel!("Norm Error")

plot([1,2,3], [4,5,6])

# savefig("Single_Precision.png")

writedlm("Cond_Numbers_Single_Precision.csv", log10.(cond_numbers), ',')
writedlm("Errors_Single_Precision.csv", errors, ',')



n, k = 100, 1
σ = 1e-10
A = almost_sing(n,k,σ)
b = randn(n)
cond(A)


#########################   Double Precision    ###########################

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
    b = randn(n)

    x1 = A \ b
    x2 = minres(A, b)
    x3 = gmres(A, b)
    x4 = rls_21(A,b,ρ)
    x5 = rls_22(A,b,ρ)

    if cond(A) >= 1e14
        push!(errors_nom_14,norm(A*x1.-b))
        push!(errors_minres_14,norm(A*x2.-b))
        push!(errors_gmres_14,norm(A*x3.-b))
        push!(errors_rob21_14,norm(A*x4.-b))
        push!(errors_rob22_14,norm(A*x5.-b))
        push!(conds_14,cond(A))
    end
    if length(conds_14) >= 1000
        break
    end
end


σ = 1e-12
for i in 1:10000
    A = almost_sing(n,k,σ)
    b = randn(n)

    x1 = A \ b
    x2 = minres(A, b)
    x3 = gmres(A, b)
    x4 = rls_21(A,b,ρ)
    x5 = rls_22(A,b,ρ)

    if cond(A) >= 1e16
        push!(errors_nom_16,norm(A*x1.-b))
        push!(errors_minres_16,norm(A*x2.-b))
        push!(errors_gmres_16,norm(A*x3.-b))
        push!(errors_rob21_16,norm(A*x4.-b))
        push!(errors_rob22_16,norm(A*x5.-b))
        push!(conds_16,cond(A))
    else
        if length(conds_15) <= 1000
            push!(errors_nom_15,norm(A*x1.-b))
            push!(errors_minres_15,norm(A*x2.-b))
            push!(errors_gmres_15,norm(A*x3.-b))
            push!(errors_rob21_15,norm(A*x4.-b))
            push!(errors_rob22_15,norm(A*x5.-b))
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

    x1 = A \ b
    x2 = minres(A, b)
    x3 = gmres(A, b)
    x4 = rls_21(A,b,ρ)
    x5 = rls_22(A,b,ρ)

    if cond(A) >= 1e18
        push!(errors_nom_18,norm(A*x1.-b))
        push!(errors_minres_18,norm(A*x2.-b))
        push!(errors_gmres_18,norm(A*x3.-b))
        push!(errors_rob21_18,norm(A*x4.-b))
        push!(errors_rob22_18,norm(A*x5.-b))
        push!(conds_18,cond(A))
    else
        if length(conds_17) <= 1000
            push!(errors_nom_17,norm(A*x1.-b))
            push!(errors_minres_17,norm(A*x2.-b))
            push!(errors_gmres_17,norm(A*x3.-b))
            push!(errors_rob21_17,norm(A*x4.-b))
            push!(errors_rob22_17,norm(A*x5.-b))
            push!(conds_17,cond(A))
        end
    end
    if (length(conds_17) >= 1000) && (length(conds_18) >= 1000)
        break
    end
end


errors_nominal = [mean(errors_nom_14),mean(errors_nom_15),mean(errors_nom_16),mean(errors_nom_17),mean(errors_nom_18)]
errors_robust = [mean(errors_rob22_14),mean(errors_rob22_15),mean(errors_rob22_16),mean(errors_rob22_17),mean(errors_rob22_18)]

cond_numbers = [1e14,1e15,1e16,1e17,1e18]
errors = hcat(errors_nominal,errors_robust)

writedlm("Cond_Numbers_Double_Precision.csv", log10.(cond_numbers), ',')
writedlm("Errors_Double_Precision.csv", errors, ',')


plot(log10.(cond_numbers), errors, title = "Double Precision Results",  label = ["Nominal" "Robust"], legend=:topleft)
xlabel!("log10(cond(A))")
ylabel!("Norm Error")


savefig("Double_Precision.png")
