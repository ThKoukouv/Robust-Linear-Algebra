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


################### Extra functions for sparsity ############################
function matrix_sparsity(B)
    n = size(B)[1]
    s = 0
    for i in 1:n
        for j in 1:n
            if B[i,j] == 0
                s += 1
            end
        end
    end
    return (s/n^2)
end

# The matrix B should be sparse and then we should add the row to make it nearly singular
function create_sparse_nsmatrix(n,k)
    B = randn(n-1,n)
    k1 = sample(1:n-1,k,replace=false)
    k2 = sample(1:n,k,replace=false)
    for i in k1
        for j in k2
            B[i,j]=0
        end
    end
    d = Normal(0.0,1e-15)
    eps1 = rand(d,n)
    A = vcat(B,sum(B[i,:] for i in 1:n-1)' .+ eps1')
    return A
end


########## Sparsity pattern for rob lin alg paper ###########


n = 100
k = 1
σ = 1e-14
ρ = 0.1
m = 5
B = randn(n,m)*1e-3*3
C = randn(n,m)*1e-3*3

sparsity, errors_minres, errors_gmres, errors_rob = [], [], [], []

vals = [31,50,71,87,95]

for v in vals
    A = create_sparse_nsmatrix(n,v)
    sp = matrix_sparsity(A)*100
    push!(sparsity,sp)
    b = randn(n)

    x2 = minres(A, b)
    x3 = gmres(A, b)
    xrob = rls_22(A,b,ρ)

    push!(errors_minres, norm(A*x2 .- b))
    push!(errors_gmres, norm(A*x3 .- b))
    push!(errors_rob, norm(A*xrob .- b))
end

errors = hcat(errors_rob,errors_minres,errors_gmres)
writedlm("RLS_Sparsity_Perc_100.csv", sparsity, ',')
writedlm("RLS_Sparsity_Errors_100.csv", errors, ',')



n = 1000
k = 1
σ = 1e-14
ρ = 0.1
m = 5
B = randn(n,m)*1e-3*3
C = randn(n,m)*1e-3*3

sparsity, errors_minres, errors_gmres, errors_rob = [], [], [], []
vals = [31,50,71,87,95]*10
for v in vals
    A = create_sparse_nsmatrix(n,v)
    sp = matrix_sparsity(A)*100
    push!(sparsity,sp)
    b = randn(n)

    x2 = minres(A, b)
    x3 = gmres(A, b)
    xrob = rls_22(A,b,ρ)

    push!(errors_minres, norm(A*x2 .- b))
    push!(errors_gmres, norm(A*x3 .- b))
    push!(errors_rob, norm(A*xrob .- b))
end

errors = hcat(errors_rob,errors_minres,errors_gmres)
writedlm("RLS_Sparsity_Perc_1000.csv", sparsity, ',')
writedlm("RLS_Sparsity_Errors_1000.csv", errors, ',')


# y = hcat(Float64.(errors_minres), Float64.(errors_gmres), Float64.(errors_rob), Float64.(errors_ad))
# plot(sparsity, y, label = ["MINRES" "GMRES" "Robust" "Adaptive"], legend=:topleft)
# title!("n=1000")
# xlabel!("Percentage of zero elements")
# ylabel!("Norm error")
# savefig("ALS_Sparsity_1000.png")
