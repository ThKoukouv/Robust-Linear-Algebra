using LinearAlgebra, JuMP, Gurobi, Random, Plots, Statistics
using DelimitedFiles, CSV, DataFrames, Distributions, StatsBase


function compute_avg_error(A,l,U)
    n = size(A)[1]
    err = mean([norm((A - l[j]*I)*U[:,j]) for j in 1:n])
    return err
end

function compute_par_avg_error(A,scenarios,l,U)
    n = size(A)[1]
    N = length(scenarios)
    errors = []
    for i in 1:n
        err = mean([norm((A + scenarios[j] - l[i]*I)*U[:,i]) for j in 1:N])
        push!(errors,err)
    end
    return mean(errors)
end


############################## Robust ###############################

function socp_first_22(A,l,r,eps,sgn)
    n = size(A)[1]
    m1 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m1, "OutputFlag", 0)
    @variable(m1, x[1:n])
    @variable(m1, u)
    @variable(m1, t)
    @constraint(m1, [u; (A - l*I)*x] in SecondOrderCone())
    @constraint(m1, [t; x] in SecondOrderCone())
    if sgn == "pos"
        @constraint(m1, sum(x[i] for i in 1:n) >= eps)
    else
        @constraint(m1, sum(x[i] for i in 1:n) <= -eps)
    end
    @objective(m1, Min, u + r*t)
    optimize!(m1)
    if termination_status(m1)  == MOI.OPTIMAL
        x_opt = JuMP.value.(x)
        return termination_status(m1), x_opt / norm(x_opt)
    else
        return [termination_status(m1)]
    end
end

function socp_solve_22(A,l,prev_eigenv,r,eps,sgn)
    n = size(A)[1]
    m = size(prev_eigenv)[2]
    m1 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m1, "OutputFlag", 0)
    @variable(m1, x[1:n])
    @variable(m1, u)
    @variable(m1, t)
    @constraint(m1, [u; (A - l*I)*x] in SecondOrderCone())
    @constraint(m1, [t; x] in SecondOrderCone())
    for i in 2:m
        @constraint(m1, x'*prev_eigenv[:,i] == 0)
    end
    if sgn == "pos"
        @constraint(m1, sum(x[i] for i in 1:n) >= eps)
    else
        @constraint(m1, sum(x[i] for i in 1:n) <= -eps)
    end
    @objective(m1, Min, u + r*t)
    optimize!(m1)
    if termination_status(m1)  == MOI.OPTIMAL
        x_opt = JuMP.value.(x)
        return termination_status(m1), x_opt / norm(x_opt)
    else
        return [termination_status(m1)]
    end
end


function socp_first_21(A,l,r,eps,sgn)
    n = size(A)[1]
    m1 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m1, "OutputFlag", 0)
    @variable(m1, x[1:n])
    @variable(m1, u)
    @variable(m1, v[1:n])

    l = [u; (A - l*I)*x]

    @constraint(m1, l in SecondOrderCone())
    @constraint(m1, x .>= -v)
    @constraint(m1, x .<= v)

    if sgn == "pos"
        @constraint(m1, sum(x[i] for i in 1:n) >= eps)
    else
        @constraint(m1, sum(x[i] for i in 1:n) <= -eps)
    end

    @objective(m1, Min, u + r*ones(n)'*v)
    optimize!(m1)
    if termination_status(m1)  == MOI.OPTIMAL
        x_opt = JuMP.value.(x)
        return termination_status(m1), x_opt / norm(x_opt)
    else
        return [termination_status(m1)]
    end
end



function socp_solve_21(A,l,prev_eigenv,r,eps,sgn)
    n = size(A)[1]
    m = size(prev_eigenv)[2]
    m1 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m1, "OutputFlag", 0)
    @variable(m1, x[1:n])
    @variable(m1, u)
    @variable(m1, v[1:n])

    l = [u; (A - l*I)*x]

    @constraint(m1, l in SecondOrderCone())
    @constraint(m1, x .>= -v)
    @constraint(m1, x .<= v)
    for i in 2:m
        @constraint(m1, x'*prev_eigenv[:,i] == 0)
    end

    if sgn == "pos"
        @constraint(m1, sum(x[i] for i in 1:n) >= eps)
    else
        @constraint(m1, sum(x[i] for i in 1:n) <= -eps)
    end

    @objective(m1, Min, u + r*ones(n)'*v)
    optimize!(m1)
    if termination_status(m1)  == MOI.OPTIMAL
        x_opt = JuMP.value.(x)
        return termination_status(m1), x_opt / norm(x_opt)
    else
        return [termination_status(m1)]
    end
end


function compute_robust_eigs1(A,l_init,r,maxiter,eps)
    n = size(A)[1]
    l = l_init
    obj_vals = []
    eig_vals = zeros(n)
    eig_vecs = zeros(n,n)
    for t in 1:maxiter
        eigen_matrix = zeros(n)
        l_new = zeros(n)
        for i in 1:n
            if i == 1
                x1 = socp_first_21(A,l[i],r,eps,"pos")
                x2 = socp_first_21(A,l[i],r,eps,"neg")
                if x1[1] != MOI.OPTIMAL && x2[1] != MOI.OPTIMAL
                    x1 = socp_first_21(A,l[i],r,eps/10,"pos")
                    x2 = socp_first_21(A,l[i],r,eps/10,"neg")
                end
            else
                x1 = socp_solve_21(A,l[i],eigen_matrix,r,eps,"pos")
                x2 = socp_solve_21(A,l[i],eigen_matrix,r,eps,"neg")
                if x1[1] != MOI.OPTIMAL && x2[1] != MOI.OPTIMAL
                    x1 = socp_solve_21(A,l[i],eigen_matrix,r,eps*1e-5,"pos")
                    x2 = socp_solve_21(A,l[i],eigen_matrix,r,eps*1e-5,"neg")
                end
            end
            if x1[1] == MOI.OPTIMAL && x2[1] == MOI.OPTIMAL
                if norm(A*x1[2] .- l[i]*x1[2]) < norm(A*x2[2] .- l[i]*x2[2])
                    x = x1[2]
                else
                    x = x2[2]
                end
            elseif x1[1] == MOI.OPTIMAL
                x = x1[2]
            else
                x = x2[2]
            end

            l_new[i] = (x'*A*x) / (x'*x)
            eigen_matrix = hcat(eigen_matrix, x)
        end
        eigenvecs = eigen_matrix[:,2:n+1]
        obj = compute_avg_error(A,l,eigenvecs)
        push!(obj_vals, obj)
        println("Iter:", t)
        eig_vals = l
        eig_vecs = eigenvecs
        l = l_new
    end
    return eig_vals, eig_vecs, obj_vals
end


function compute_robust_eigs2(A,l_init,r,maxiter,eps)
    n = size(A)[1]
    l = l_init
    obj_vals = []
    eig_vals = zeros(n)
    eig_vecs = zeros(n,n)
    for t in 1:maxiter
        eigen_matrix = zeros(n)
        l_new = zeros(n)
        for i in 1:n
            if i == 1
                x1 = socp_first_22(A,l[i],r,eps,"pos")
                x2 = socp_first_22(A,l[i],r,eps,"neg")
                if x1[1] != MOI.OPTIMAL && x2[1] != MOI.OPTIMAL
                    x1 = socp_first_22(A,l[i],r,eps/10,"pos")
                    x2 = socp_first_22(A,l[i],r,eps/10,"neg")
                end
            else
                x1 = socp_solve_22(A,l[i],eigen_matrix,r,eps,"pos")
                x2 = socp_solve_22(A,l[i],eigen_matrix,r,eps,"neg")
                if x1[1] != MOI.OPTIMAL && x2[1] != MOI.OPTIMAL
                    x1 = socp_solve_22(A,l[i],eigen_matrix,r,eps*1e-5,"pos")
                    x2 = socp_solve_22(A,l[i],eigen_matrix,r,eps*1e-5,"neg")
                end
            end
            if x1[1] == MOI.OPTIMAL && x2[1] == MOI.OPTIMAL
                if norm(A*x1[2] .- l[i]*x1[2]) < norm(A*x2[2] .- l[i]*x2[2])
                    x = x1[2]
                else
                    x = x2[2]
                end
            elseif x1[1] == MOI.OPTIMAL
                x = x1[2]
            else
                x = x2[2]
            end

            l_new[i] = (x'*A*x) / (x'*x)
            eigen_matrix = hcat(eigen_matrix, x)
        end
        eigenvecs = eigen_matrix[:,2:n+1]
        obj = compute_avg_error(A,l,eigenvecs)
        push!(obj_vals, obj)
        println("Iter:", t)
        eig_vals = l
        eig_vecs = eigenvecs
        l = l_new
    end
    return eig_vals, eig_vecs, obj_vals
end

# Function for calculating the percentage of zero eigenvalues
function calculate_perc_zeros(x)
    n = size(x)[1]
    p = sum(x .<= 1e-5)
    return (p/n)*100
end





###################   Main Experiments    ########################

n = 10
ρ =  0.1
errors_nominal, errors_robust_l1, errors_robust_l2 = [], [], []
cond_numbs, zero_eigs = [], []
k_vals = [10,0,-3,-5,-8]

for k in k_vals
    errors_nom, errors_rob_l1, errors_rob_l2 = [], [], []
    conds, zero_eigen_perc  = [], []
    for i in 1:10
        C = randn(n+k,n)
        A = C'*C

        l_nom, u_nom = eigen(A)
        l_ro1, u_ro1, obj_vals1 = compute_robust_eigs1(A, l_nom, ρ, 20, 0.001)
        l_ro2, u_ro2, obj_vals2 = compute_robust_eigs2(A, l_nom, ρ, 20, 0.001)

        l_nom, u_nom = eigen(A)
        errors_nom_aux, errors_rob_l2_aux, errors_rob_l1_aux = [], [], []
        total_vals = collect(15:0.01:25)
        pert_vals = sample(total_vals,100)
        for p in pert_vals
            DA = rand(n,n)*p
            for i in 1:n
                for j in 1:n
                    DA[i,j] = DA[j,i]
                end
            end
            B = A + DA
            push!(errors_nom_aux, compute_avg_error(B, l_nom, u_nom))
            push!(errors_rob_l2_aux, compute_avg_error(B, l_ro2, u_ro2))
            push!(errors_rob_l1_aux, compute_avg_error(B, l_ro1, u_ro1))
        end
        push!(errors_nom, mean(errors_nom_aux))
        push!(errors_rob_l2, mean(errors_rob_l2_aux))
        push!(errors_rob_l1, mean(errors_rob_l1_aux))
        push!(conds, cond(A))
        push!(zero_eigen_perc,calculate_perc_zeros(l_nom))
    end
    push!(errors_nominal,mean(errors_nom))
    push!(errors_robust_l1,mean(errors_rob_l1))
    push!(errors_robust_l2,mean(errors_rob_l2))
    push!(cond_numbs,mean(conds))
    push!(zero_eigs,mean(zero_eigen_perc))
end

errors_both = hcat(errors_nominal,errors_robust_l1)

writedlm("Cond_Numbers_Eigs.csv", log10.(cond_numbs[2:end]), ',')
writedlm("Errors_Eigs.csv", errors_both[2:end,:], ',')


plot(log10.(cond_numbs[2:end]), errors_both[2:end,:], title = "Nominal vs Robust Eigenvectors",  label = ["Nominal" "Robust"], legend=:bottomleft)
xlabel!("log10(cond(A))")
ylabel!("Norm Error")
savefig("Eigen_CondNumber_Effect.png")


# Experiments for Table in eigenvalue section

n = 10
ρ = 0.5
# n = 10, small perturbations
errors_nom, errors_rob_l1, errors_rob_l2, pert_norm = [], [], [], []
for i in 1:10
    C = randn(n-1,n)
    A = C'*C

    l_nom, u_nom = eigen(A)
    l_ro1, u_ro1, obj_vals1 = compute_robust_eigs1(A, l_nom, ρ, 20, 0.001)
    l_ro2, u_ro2, obj_vals2 = compute_robust_eigs2(A, l_nom, ρ, 20, 0.001)

    l_nom, u_nom = eigen(A)
    errors_nom_aux, errors_rob_l2_aux, errors_rob_l1_aux, pert_norm_aux = [], [], [], []
    total_vals = collect(5:0.01:10)
    pert_vals = sample(total_vals,100)
    for p in pert_vals
        DA = rand(n,n)*p
        for i in 1:n
            for j in 1:n
                DA[i,j] = DA[j,i]
            end
        end
        B = A + DA
        push!(errors_nom_aux, compute_avg_error(B, l_nom, u_nom))
        push!(errors_rob_l2_aux, compute_avg_error(B, l_ro2, u_ro2))
        push!(errors_rob_l1_aux, compute_avg_error(B, l_ro1, u_ro1))
        push!(pert_norm_aux, norm(DA))
    end
    push!(errors_nom, mean(errors_nom_aux))
    push!(errors_rob_l2, mean(errors_rob_l2_aux))
    push!(errors_rob_l1, mean(errors_rob_l1_aux))
    push!(pert_norm, mean(pert_norm_aux))
end

println(mean(pert_norm))
println(std(pert_norm))
println(mean(errors_nom))
println(std(errors_nom))
println(mean(errors_rob_l1))
println(std(errors_rob_l1))
println(mean(errors_rob_l2))
println(std(errors_rob_l2))

# n = 10, large perturbations
errors_nom, errors_rob_l1, errors_rob_l2, pert_norm = [], [], [], []
for i in 1:10
    C = randn(n-1,n)
    A = C'*C

    l_nom, u_nom = eigen(A)
    l_ro1, u_ro1, obj_vals1 = compute_robust_eigs1(A, l_nom, ρ, 20, 0.001)
    l_ro2, u_ro2, obj_vals2 = compute_robust_eigs2(A, l_nom, ρ, 20, 0.001)

    l_nom, u_nom = eigen(A)
    errors_nom_aux, errors_rob_l2_aux, errors_rob_l1_aux, pert_norm_aux = [], [], [], []
    total_vals = collect(15:0.01:25)
    pert_vals = sample(total_vals,100)
    for p in pert_vals
        DA = rand(n,n)*p
        for i in 1:n
            for j in 1:n
                DA[i,j] = DA[j,i]
            end
        end
        B = A + DA
        push!(errors_nom_aux, compute_avg_error(B, l_nom, u_nom))
        push!(errors_rob_l2_aux, compute_avg_error(B, l_ro2, u_ro2))
        push!(errors_rob_l1_aux, compute_avg_error(B, l_ro1, u_ro1))
        push!(pert_norm_aux, norm(DA))
    end
    push!(errors_nom, mean(errors_nom_aux))
    push!(errors_rob_l2, mean(errors_rob_l2_aux))
    push!(errors_rob_l1, mean(errors_rob_l1_aux))
    push!(pert_norm, mean(pert_norm_aux))
end

println(mean(pert_norm))
println(std(pert_norm))
println(mean(errors_nom))
println(std(errors_nom))
println(mean(errors_rob_l1))
println(std(errors_rob_l1))
println(mean(errors_rob_l2))
println(std(errors_rob_l2))


n = 100
# n= 100, small perturbations
errors_nom, errors_rob_l1, errors_rob_l2, pert_norm = [], [], [], []
for i in 1:10
    C = randn(n-1,n)
    A = C'*C

    l_nom, u_nom = eigen(A)
    l_ro1, u_ro1, obj_vals1 = compute_robust_eigs1(A, l_nom, ρ, 30, 0.001)
    l_ro2, u_ro2, obj_vals2 = compute_robust_eigs2(A, l_nom, ρ, 30, 0.001)

    l_nom, u_nom = eigen(A)
    errors_nom_aux, errors_rob_l2_aux, errors_rob_l1_aux, pert_norm_aux = [], [], [], []
    total_vals = collect(5:0.01:10)
    pert_vals = sample(total_vals,100)
    for p in pert_vals
        DA = rand(n,n)*p
        for i in 1:n
            for j in 1:n
                DA[i,j] = DA[j,i]
            end
        end
        B = A + DA
        push!(errors_nom_aux, compute_avg_error(B, l_nom, u_nom))
        push!(errors_rob_l2_aux, compute_avg_error(B, l_ro2, u_ro2))
        push!(errors_rob_l1_aux, compute_avg_error(B, l_ro1, u_ro1))
        push!(pert_norm_aux, norm(DA))
    end
    push!(errors_nom, mean(errors_nom_aux))
    push!(errors_rob_l2, mean(errors_rob_l2_aux))
    push!(errors_rob_l1, mean(errors_rob_l1_aux))
    push!(pert_norm, mean(pert_norm_aux))
end

println(mean(pert_norm))
println(std(pert_norm))
println(mean(errors_nom))
println(std(errors_nom))
println(mean(errors_rob_l1))
println(std(errors_rob_l1))
println(mean(errors_rob_l2))
println(std(errors_rob_l2))


# n=100, large perturbations
errors_nom, errors_rob_l1, errors_rob_l2, pert_norm = [], [], [], []
for i in 1:10
    C = randn(n-1,n)
    A = C'*C

    l_nom, u_nom = eigen(A)
    l_ro1, u_ro1, obj_vals1 = compute_robust_eigs1(A, l_nom, ρ, 30, 0.001)
    l_ro2, u_ro2, obj_vals2 = compute_robust_eigs2(A, l_nom, ρ, 30, 0.001)

    l_nom, u_nom = eigen(A)
    errors_nom_aux, errors_rob_l2_aux, errors_rob_l1_aux, pert_norm_aux = [], [], [], []
    total_vals = collect(15:0.01:25)
    pert_vals = sample(total_vals,100)
    for p in pert_vals
        DA = rand(n,n)*p
        for i in 1:n
            for j in 1:n
                DA[i,j] = DA[j,i]
            end
        end
        B = A + DA
        push!(errors_nom_aux, compute_avg_error(B, l_nom, u_nom))
        push!(errors_rob_l2_aux, compute_avg_error(B, l_ro2, u_ro2))
        push!(errors_rob_l1_aux, compute_avg_error(B, l_ro1, u_ro1))
        push!(pert_norm_aux, norm(DA))
    end
    push!(errors_nom, mean(errors_nom_aux))
    push!(errors_rob_l2, mean(errors_rob_l2_aux))
    push!(errors_rob_l1, mean(errors_rob_l1_aux))
    push!(pert_norm, mean(pert_norm_aux))
end

println(mean(pert_norm))
println(std(pert_norm))
println(mean(errors_nom))
println(std(errors_nom))
println(mean(errors_rob_l1))
println(std(errors_rob_l1))
println(mean(errors_rob_l2))
println(std(errors_rob_l2))




################   Computational Times  for Eiegnevalues   #################

n = 10
ρ = 0.1
times_nom_10, times_rob21_10, times_rob22_10 = [], [], []
for i in 1:100
    C = randn(n,n)
    A = C'*C

    t1 = time_ns()
    l_nom, u_nom = eigen(A)
    t2 = time_ns()
    total_time_nom = (t2-t1)*10^(-9)

    t1 = time_ns()
    l_ro1, u_ro1, obj_vals1 = compute_robust_eigs1(A, l_nom, ρ, 20, 0.001)
    t2 = time_ns()
    total_time_rob21 = (t2-t1)*10^(-9)

    t1 = time_ns()
    l_ro2, u_ro2, obj_vals2 = compute_robust_eigs2(A, l_nom, ρ, 20, 0.001)
    t2 = time_ns()
    total_time_rob22 = (t2-t1)*10^(-9)


    push!(times_nom_10, total_time_nom)
    push!(times_rob21_10, total_time_rob21)
    push!(times_rob22_10, total_time_rob22)
end

n = 100
ρ = 0.1
times_nom_100, times_rob21_100, times_rob22_100 = [], [], []
for i in 1:10
    C = randn(n,n)
    A = C'*C

    t1 = time_ns()
    l_nom, u_nom = eigen(A)
    t2 = time_ns()
    total_time_nom = (t2-t1)*10^(-9)

    t1 = time_ns()
    l_ro1, u_ro1, obj_vals1 = compute_robust_eigs1(A, l_nom, ρ, 30, 0.001)
    t2 = time_ns()
    total_time_rob21 = (t2-t1)*10^(-9)

    t1 = time_ns()
    l_ro2, u_ro2, obj_vals2 = compute_robust_eigs2(A, l_nom, ρ, 30, 0.001)
    t2 = time_ns()
    total_time_rob22 = (t2-t1)*10^(-9)


    push!(times_nom_100, total_time_nom)
    push!(times_rob21_100, total_time_rob21)
    push!(times_rob22_100, total_time_rob22)
end



n = 300
ρ = 0.1
times_nom_300, times_rob21_300, times_rob22_300 = [], [], []
for i in 1:1
    C = randn(n,n)
    A = C'*C

    t1 = time_ns()
    l_nom, u_nom = eigen(A)
    t2 = time_ns()
    total_time_nom = (t2-t1)*10^(-9)

    t1 = time_ns()
    l_ro1, u_ro1, obj_vals1 = compute_robust_eigs1(A, l_nom, ρ, 30, 0.0001)
    t2 = time_ns()
    total_time_rob21 = (t2-t1)*10^(-9)

    t1 = time_ns()
    l_ro2, u_ro2, obj_vals2 = compute_robust_eigs2(A, l_nom, ρ, 30, 0.0001)
    t2 = time_ns()
    total_time_rob22 = (t2-t1)*10^(-9)

    push!(times_nom_300, total_time_nom)
    push!(times_rob21_300, total_time_rob21)
    push!(times_rob22_300, total_time_rob22)
end

println("n=10")
println("Nominal:")
println(mean(times_nom_10))
println(std(times_nom_10))
println("Robust 21:")
println(mean(times_rob21_10))
println(std(times_rob21_10))
println("Robust 22:")
println(mean(times_rob22_10))
println(std(times_rob22_10))


println("n=100")
println("Nominal:")
println(mean(times_nom_100))
println(std(times_nom_100))
println("Robust 21:")
println(mean(times_rob21_100))
println(std(times_rob21_100))
println("Robust 22:")
println(mean(times_rob22_100))
println(std(times_rob22_100))


println("n=300")
println("Nominal:")
println(mean(times_nom_300))
println(std(times_nom_300))
println("Robust 21:")
println(mean(times_rob21_300))
println(std(times_rob21_300))
println("Robust 22:")
println(mean(times_rob22_300))
println(std(times_rob22_300))
