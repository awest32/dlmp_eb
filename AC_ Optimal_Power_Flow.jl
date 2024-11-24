#### AC Optimal Power Flow ####

# This file provides a pedagogical example of modeling the AC Optimal Power
# Flow problem using the Julia Mathematical Programming package (JuMP) and the
# PowerModels package for data parsing.

# This file can be run by calling `include("ac-opf.jl")` from the Julia REPL or
# by calling `julia ac-opf.jl` in Julia v1.

# Developed by Line Roald (@lroald) and Carleton Coffrin (@ccoffrin)


###############################################################################
# 0. Initialization
###############################################################################

# Load Julia Packages
#--------------------
using PowerModels
using Ipopt
using JuMP
using PowerPlots
using PowerModelsAnalytics
using CSV, Plots
import GR
using Dates 
using DataFrames
using LinearAlgebra
using Printf
using JLD2
include("index.jl")
# Load System Data
# ----------------
powermodels_path = joinpath(dirname(pathof(PowerModels)), "../")
network = "case33bw_aw_edit" #"case5_matlab"#"case33bw"#"RTS_GMLC"#"case57"#"case118"#pglib_opf_case240_pserc"
units = "MW"
# if network == "case33bw"
#     units == "kW"
# else
#     units == "MW"
# end
if network != "case33bw_aw_edit"
    file_name = "$(powermodels_path)/test/data/matpower/$network.m"
else
    file_name = "$network.m"
end
# note: change this string to modify the network data that will be loaded

# load the data file
data = PowerModels.parse_file(file_name)
powerplot(data)

#get the date 
date = Dates.today()

plot_types = ["lmp", "line_flows", "voltage_angles", "voltage_magnitudes", "line_flows_and_limits", "voltage_magnitudes_and_limits", "percent_loadserved", "gini", "jain", "current_magnitudes_and_limits"]
#make a save folder for the plots 
if !isdir("plots_$date")
    mkdir("plots_$date")
end
for plot_type in plot_types
        #println("plot_$date/$plot_type")
        if !isdir("plots_$date/$plot_type")
            mkdir("plots_$date/$plot_type")
        end
end
save_folder = "plots_$date/"

# Define the number of tests and scenarios (combinations of 4-bit logic)
n_tests = 10
n_scenarios = 16  # 2^4 = 16 combinations of 4-bit boolean flags
scenarios = ["ls_obj", "ls_fair_obj_1","ls_fair_obj_2"]#["ls_obj","ls_fair_obj_1","ls_fair_obj_2"]#"ls_fair_constr"]
# Now, generate a 2D matrix of boolean flags for the tests
# Each row represents a test, and columns represent the scenarios
boolean_flags = Bool[true false false; true true false; true false  true]

#println("\nBoolean flags for each test (10x16 matrix):")
#println(boolean_flags)

# Define the cost of load shedding
c_load_vary = 2e3#LinRange(200, 2000, 2), 200 same as gen cost so shed load, 2000 too high no load shed
# Assume the cost of varying the load is the same as the cost of generaxting power

# Define the cost of fairness in load shedding 
c_fair = LinRange(1, 1E2, 2)#*1e2

# Define a parameter to scale the load
# Increasing will create an undervoltage problem
scale_load = 10.0
scale_gen = scale_load

# Define the threshold for the energy burden, aka the energy poverty threshold
eb_threshold = 0.06 # 6% of the average load
dlmp_base = rand(0.5:0.01:1.5, length(data["load"]))*.2 #.2 $/kWh

# Add zeros to turn linear objective functions into quadratic ones
# so that additional parameter checks are not required
PowerModels.standardize_cost_terms!(data, order=2)

# Adds reasonable rate_a values to branches without them
PowerModels.calc_thermal_limits!(data)

# use build_ref to filter out inactive components
ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
# note: ref contains all the relevant system parameters needed to build the OPF model
# When we introduce constraints and variable bounds below, we use the parameters in ref.

###############################################################################
# 1. Building the Optimal Power Flow Model
###############################################################################

# Initialize a JuMP Optimization Model
#-------------------------------------
model = Model(Ipopt.Optimizer)

set_optimizer_attribute(model, "print_level", 0)
# note: print_level changes the amount of solver information printed to the terminal


# Add Optimization and State Variables
# ------------------------------------
model.ext[:variables] = Dict()
# Add voltage angles va for each bus
bus_va = model.ext[:variables][:bus_va] = @variable(model, va[i in keys(ref[:bus])])
# note: [i in keys(ref[:bus])] adds one `va` variable for each bus in the network
#ref[:bus][i]["vmin"]
# Add voltage angles vm for each bus
bus_vm = model.ext[:variables][:bus_vm] = @variable(model,  0.95<= vm[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"], start=1.0)
# note: this vairable also includes the voltage magnitude limits and a starting value

# Add active power generation variable pg for each generator (including limits)
pg = model.ext[:variables][:pg] = @variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= scale_gen*ref[:gen][i]["pmax"])
# Add reactive power generation variable qg for each generator (including limits)
qg = model.ext[:variables][:qg] = @variable(model, ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= scale_gen*ref[:gen][i]["qmax"])

# Add power flow variables p to represent the active power flow for each branch
pf = model.ext[:variables][:pf] = @variable(model, -ref[:branch][l]["rate_a"] <= p[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])
# Add power flow variables q to represent the reactive power flow for each branch
qf = model.ext[:variables][:qf] = @variable(model, -ref[:branch][l]["rate_a"] <= q[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])
# note: ref[:arcs] includes both the from (i,j) and the to (j,i) sides of a branch

# Add power flow variables p_dc to represent the active power flow for each HVDC line
p_dc = model.ext[:variables][:p_dc] = @variable(model, p_dc[a in ref[:arcs_dc]])
# Add power flow variables q_dc to represent the reactive power flow at each HVDC terminal
q_dc = model.ext[:variables][:q_dc] = @variable(model, q_dc[a in ref[:arcs_dc]])

# Add a variable to represent the cost of active power load at each bus (AW added 9/17/2024)
# Find max load out of all of the loads
max_load = scale_load*maximum([load["pd"] for (i,load) in ref[:load]])
min_load = 0.1*max_load #minimum([load["pd"] for (i,load) in ref[:load]])
p_load = model.ext[:variables][:p_load] = @variable(model, min_load <= p_load[i in keys(ref[:load])] <= max_load)

for i in keys(ref[:load])
    @constraint(model, p_load[i]  .<= scale_load*ref[:load][i]["pd"])
end
for (l,dcline) in ref[:dcline]
    f_idx = (l, dcline["f_bus"], dcline["t_bus"])
    t_idx = (l, dcline["t_bus"], dcline["f_bus"])

    JuMP.set_lower_bound(p_dc[f_idx], dcline["pminf"])
    JuMP.set_upper_bound(p_dc[f_idx], dcline["pmaxf"])
    JuMP.set_lower_bound(q_dc[f_idx], dcline["qminf"])
    JuMP.set_upper_bound(q_dc[f_idx], dcline["qmaxf"])

    JuMP.set_lower_bound(p_dc[t_idx], dcline["pmint"])
    JuMP.set_upper_bound(p_dc[t_idx], dcline["pmaxt"])
    JuMP.set_lower_bound(q_dc[f_idx], dcline["qmint"])
    JuMP.set_upper_bound(q_dc[f_idx], dcline["qmaxt"])
end

# Add Constraints
# ---------------
model.ext[:constraints] = Dict()
model.ext[:constraints][:nodal_active_power_balance] = []
# Fix the voltage angle to zero at the reference bus
for (i,bus) in ref[:ref_buses]
model.ext[:constraints][:va_i] = @constraint(model, va[i] == 0)
end

# Add the energy burden constraint
# model.ext[:variables][:energy_burdens] = @variable(model, eb[i in keys(ref[:load])]>=0)
# model.ext[:constraints][:energy_burden] = @constraint(model, eb .<= eb_threshold)
# model.ext[:constraints][:energy_burden] = @constraint(model, eb .== [load["pd"] for (i,load) in ref[:load]]./dlmp_base)

# Nodal power balance constraints
for (i,bus) in ref[:bus]
    # Build a list of the loads and shunt elements connected to the bus i
    bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
    #print("Bus loads: ", bus_loads[i])
    bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

    # Active power balance at node i
    # AW updated to extract nodal active power balance constraints in a vector
    push!(model.ext[:constraints][:nodal_active_power_balance],@constraint(model,
        sum(p[a] for a in ref[:bus_arcs][i]) +                  # sum of active power flow on lines from bus i +
        sum(p_dc[a_dc] for a_dc in ref[:bus_arcs_dc][i]) ==     # sum of active power flow on HVDC lines from bus i =
        sum(pg[g] for g in ref[:bus_gens][i]) -                 # sum of active power generation at bus i -
        #sum(load["pd"] for load in bus_loads) -                 # sum of active load consumption at bus i -
        sum(p_load[p_l] for p_l in ref[:bus_loads][i]) -                 # sum of active load consumption at bus i -
        sum(shunt["gs"] for shunt in bus_shunts)*vm[i]^2        # sum of active shunt element injections at bus i
    ))

    # Reactive power balance at node i
    model.ext[:constraints][:nodal_reactive_power_balance] = @constraint(model,
        sum(q[a] for a in ref[:bus_arcs][i]) +                  # sum of reactive power flow on lines from bus i +
        sum(q_dc[a_dc] for a_dc in ref[:bus_arcs_dc][i]) ==     # sum of reactive power flow on HVDC lines from bus i =
        sum(qg[g] for g in ref[:bus_gens][i]) -                 # sum of reactive power generation at bus i -
        sum(load["qd"] for load in bus_loads) +                 # sum of reactive load consumption at bus i -
        sum(shunt["bs"] for shunt in bus_shunts)*vm[i]^2        # sum of reactive shunt element injections at bus i
    )

    ########
    #AW added slack bus voltage magnitude equality constraint
    #9/13/2024
    # Bus voltage magnitude limit for slack bus
    if ref[:bus][i]["bus_type"] == 3
        slack_vm_limit = model.ext[:constraints][:slack_vm_limits] = @constraint(model, vm[i] == 1)
    end

end

# Branch power flow physics and limit constraints
#print("Branches: ", ref[:branch])
branch_id = collect(keys(ref[:branch]))
# for i in branch_id
#     println("Branch maximum limits $i rate b: ", ref[:branch][i]["rate_a"])
# end

for (i,branch) in ref[:branch]
    # Build the from variable id of the i-th branch, which is a tuple given by (branch id, from bus, to bus)
    f_idx = (i, branch["f_bus"], branch["t_bus"])
    # Build the to variable id of the i-th branch, which is a tuple given by (branch id, to bus, from bus)
    t_idx = (i, branch["t_bus"], branch["f_bus"])
    # note: it is necessary to distinguish between the from and to sides of a branch due to power losses

    p_fr = p[f_idx]                     # p_fr is a reference to the optimization variable p[f_idx]
    q_fr = q[f_idx]                     # q_fr is a reference to the optimization variable q[f_idx]
    p_to = p[t_idx]                     # p_to is a reference to the optimization variable p[t_idx]
    q_to = q[t_idx]                     # q_to is a reference to the optimization variable q[t_idx]
    # note: adding constraints to p_fr is equivalent to adding constraints to p[f_idx], and so on

    vm_fr = vm[branch["f_bus"]]         # vm_fr is a reference to the optimization variable vm on the from side of the branch
    vm_to = vm[branch["t_bus"]]         # vm_to is a reference to the optimization variable vm on the to side of the branch
    va_fr = va[branch["f_bus"]]         # va_fr is a reference to the optimization variable va on the from side of the branch
    va_to = va[branch["t_bus"]]         # va_fr is a reference to the optimization variable va on the to side of the branch

    # Compute the branch parameters and transformer ratios from the data
    g, b = PowerModels.calc_branch_y(branch)
    tr, ti = PowerModels.calc_branch_t(branch)
    g_fr = branch["g_fr"]
    b_fr = branch["b_fr"]
    g_to = branch["g_to"]
    b_to = branch["b_to"]
    tm = branch["tap"]^2
    # note: tap is assumed to be 1.0 on non-transformer branches


    # AC Power Flow Constraints

    # From side of the branch flow
    model.ext[:constraints][:branch_flow_p_fr] = @constraint(model, p_fr ==  (g+g_fr)/tm*vm_fr^2 + (-g*tr+b*ti)/tm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/tm*(vm_fr*vm_to*sin(va_fr-va_to)) )
    model.ext[:constraints][:branch_flow_q_fr] =  @constraint(model, q_fr == -(b+b_fr)/tm*vm_fr^2 - (-b*tr-g*ti)/tm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/tm*(vm_fr*vm_to*sin(va_fr-va_to)) )

    # To side of the branch flow
    model.ext[:constraints][:branch_flow_p_to] = @constraint(model, p_to ==  (g+g_to)*vm_to^2 + (-g*tr-b*ti)/tm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/tm*(vm_to*vm_fr*sin(va_to-va_fr)) )
    model.ext[:constraints][:branch_flow_q_to] = @constraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/tm*(vm_to*vm_fr*cos(va_fr-va_to)) + (-g*tr-b*ti)/tm*(vm_to*vm_fr*sin(va_to-va_fr)) )

    # Voltage angle difference limit
    model.ext[:constraints][:va_diff_max_lim] = @constraint(model, va_fr - va_to <= branch["angmax"])
    model.ext[:constraints][:va_diff_min_lim] = @constraint(model, va_fr - va_to >= branch["angmin"])

    # Apparent power limit, from side and to side
    model.ext[:constraints][:apparent_power_balance_fr] = @constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
    model.ext[:constraints][:apparent_power_balance_to] = @constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
end

# HVDC line constraints
for (i,dcline) in ref[:dcline]
    # Build the from variable id of the i-th HVDC line, which is a tuple given by (hvdc line id, from bus, to bus)
    f_idx = (i, dcline["f_bus"], dcline["t_bus"])
    # Build the to variable id of the i-th HVDC line, which is a tuple given by (hvdc line id, to bus, from bus)
    t_idx = (i, dcline["t_bus"], dcline["f_bus"])   # index of the ith HVDC line which is a tuple given by (line number, to bus, from bus)
    # note: it is necessary to distinguish between the from and to sides of a HVDC line due to power losses

    # Constraint defining the power flow and losses over the HVDC line
    model.ext[:constraints][:hvdc] = @constraint(model, (1-dcline["loss1"])*p_dc[f_idx] + (p_dc[t_idx] - dcline["loss0"]) == 0)
end


# Add Objective Function
# ----------------------

# index representing which side the HVDC line is starting
from_idx = Dict(arc[1] => arc for arc in ref[:arcs_from_dc])

# Save the load shedding cost and load shedding amount
ls_dict = DataFrame(ls_cost=Float32[], ls_amount=Float32[])
ls_amount_vec = []
ls_percent_dict = Dict()
lmp_per_loadshed = Dict()
gini_per_loadshed = []
jain_per_loadshed = []
total_load =[]
network_buses = keys(ref[:bus])

net_buses = collect(network_buses)

results = Dict()
# Create the DataFrame
dict = Dict(
    "gini" => 0,
    "jain" => 0,
    "ls_percent" => zeros(length(net_buses)),  # Vectors are stored as single entries in rows
    "vm" => zeros(length(net_buses)),
    "line_flow_percent" => zeros(length(ref[:branch])),
    "im" => zeros(length(ref[:branch])),
    "im_max" => zeros(length(ref[:branch])),
    "im_min" => zeros(length(ref[:branch])),
    "lmp" => zeros(length(net_buses))
)

# Populate the nested dictionary structure
(r,c) = size(boolean_flags)
objective_choice = range(1, r, step=1)
for i in objective_choice
    results[i] = Dict()
    for j in c_fair
        results[i][j] = Dict()
        for k in c_load_vary
            results[i][j][k] = dict
        end
    end
end


# BAU AC OPF objective function
# Minimize the cost of active power generation and cost of HVDC line usage
@expression(model, acopf_obj, sum(gen["cost"][1]*pg[i]^2 + gen["cost"][2]*pg[i] + gen["cost"][3] for (i,gen) in ref[:gen]) +
sum(dcline["cost"][1]*p_dc[from_idx[i]]^2 + dcline["cost"][2]*p_dc[from_idx[i]] + dcline["cost"][3] for (i,dcline) in ref[:dcline])
)

@expression(model, load_shed_obj,sum(c_load_vary*(scale_load*load["pd"] - p_load[i]) for (i,load) in ref[:load]))
################################################################################################
# 2. Conduct the experiments
################################################################################################
# Add a variable to determine the best distane from the average for load shedding. Tries to keep the load shedding  equality
model.ext[:variables][:load_shedding_fairness] = @variable(model, c_ls_fair>=0.0001)

#################################################################################################
# Loop through all the tests defined in the experimental setup
#################################################################################################
(r,c) = size(boolean_flags)
total_percent_load_served = []
for i in 1:r
    flags = boolean_flags[i,:]
    #println("Test $flags")
    true_flags = findall(flags)
    #println("True flags: $true_flags")
    naming_test_files = ""
    if flags[2] == true
        cost_fair = c_fair
        fair_obj_1 = 1
    else
        fair_obj_1 = 0
    end
    if flags[3] == true
        cost_fair = c_fair
        fair_obj_2 = 1
    else
        fair_obj_2 = 0
    end

    # Determine which scenarios are active
    println("Scenarios: $scenarios")
    println("Active Test Scenarios: $(scenarios[true_flags])")

    for cost_fair in c_fair
        #for cost_ls in c_load_vary

            # Minimize the cost of active power generation and cost of HVDC line usage
            # assumes costs are given as quadratic functions
            @objective(model, Min, acopf_obj
                            + load_shed_obj
                #+ fair_obj_1*cost_fair*sum(abs(c_ls_fair-(p_load[i]/(scale_load*load["pd"]))) for (i,load) in ref[:load])
                #+ fair_obj_2*cost_fair*sum(abs((p_load[i+1]/(scale_load*ref[:load][i+1]["pd"])) - (p_load[i]/(scale_load*ref[:load][i]["pd"]))) for i in 1:(length(ref[:load])-1))
            )

            ###############################################################################
            # 3. Solve the Optimal Power Flow Model and Review the Results
            ###############################################################################

            # Solve the optimization problem
            optimize!(model)

            # Check that the solver terminated without an error
            println("The solver termination status is $(termination_status(model))")

            # Print the cost of fairness
            println("THE COST OF FAIRNESS IS $(cost_fair) (USD).")

            # Check the value of the objective function
            cost = objective_value(model)
            println("The cost of generation is $(cost) (USD).")

            # print the total load
            println("The total load is $(scale_load*sum([load["pd"] for (i,load) in ref[:load]])*ref[:baseMVA]) $units.")

            #print the total active load
            println("The total active load is $(sum([value(p_load[i]) for (i,load) in ref[:load]])*ref[:baseMVA]) $units.")
            
            #print the total generation
            println("The total generation is $(sum([value(pg[i]) for (i,gen) in ref[:gen]])*ref[:baseMVA]) $units.")
            
            #print the total generation capacity
            println("The total generation capacity is $(scale_gen*sum([gen["pmax"] for (i,gen) in ref[:gen]])*ref[:baseMVA]) $units.")
            
            #print the minimum voltage magnitude of the network
            println("The minimum voltage magnitude is $(minimum([value(vm[i]) for (i,bus) in ref[:bus]])) p.u.")
            
            total_ref_load = scale_load*sum([load["pd"] for (i,load) in ref[:load]])

            ls_amount_vec = []
            for (i,load) in ref[:load]
                push!(ls_amount_vec, abs(value(p_load[i]) - scale_load*ref[:load][i]["pd"]))
            end
            println("The total load shed is $(sum(ls_amount_vec)*ref[:baseMVA]) $units.")
            ls_amount_total  =  sum(ls_amount_vec)

            load_served_amount_vec = []
            for (i,load) in ref[:load]
                push!(load_served_amount_vec, value(p_load[i]))
            end
            println("The total load served is $(sum(load_served_amount_vec)*ref[:baseMVA]) $units.")
            
            load_served_percent_per_node = []
            for (i,load) in ref[:load]
                push!(load_served_percent_per_node, value(p_load[i])/(scale_load*ref[:load][i]["pd"]))
            end
            #AW update 9/23/2024
            ##### remove zero load values in the ls_amount_vec
            load_served_percent_per_node = load_served_percent_per_node[load_served_percent_per_node .!= 0]
            results[i][cost_fair][c_load_vary]["ls_percent"] = load_served_percent_per_node*100
            #println("Load Served Percent: ", (results[i][cost_fair][c_load_vary]["ls_percent"]))

            #print the average percentage of load served 
            println("The total percentage of load served is $((sum(load_served_amount_vec)/(total_ref_load))*100) %.")

            #print the total load served
            push!(total_percent_load_served, (sum(load_served_amount_vec)/(total_ref_load))*100)

            total_load = scale_load * sum([load["pd"] for (i,load) in ref[:load]])
            loadshed_percent_vec = []
            for (i,load) in ref[:load]
                push!(loadshed_percent_vec, abs(scale_load*ref[:load][i]["pd"] - value(p_load[i]) )/(scale_load*ref[:load][i]["pd"]))
            end
            
            # calculate the Gini coefficient for the load shedding
            results[i][cost_fair][c_load_vary]["gini"] = gini_coefficient(load_served_percent_per_node)

            # calculate the Jain index for the load shedding
            results[i][cost_fair][c_load_vary]["jain"] = jain(load_served_percent_per_node)
                    
            #print the gini and jain index
            println("The Gini coefficient is $(results[i][cost_fair][c_load_vary]["gini"]).")
            println("The Jain index is $(results[i][cost_fair][c_load_vary]["jain"]).")

            # save the active power lmps
            active_power_nodal_constraints = Dict(network_buses .=> model.ext[:constraints][:nodal_active_power_balance])
            lmp_vec = []
            for (i,bus) in ref[:bus]
                push!(lmp_vec,dual(active_power_nodal_constraints[i]))
            end


            results[i][cost_fair][c_load_vary]["lmp"] = lmp_vec
            # This is the lmp in dollars per MW
            # divide by 1000 to get it in dollars per KW
            println("The LMPs are $(-lmp_vec/10^3).")
            println("The average of the DLMPs are $(sum(-lmp_vec/10^3)/length(lmp_vec)).")
            # Save the voltage magnitudes
            voltage_magnitudes = []
            for (i,bus) in ref[:bus]
                push!(voltage_magnitudes,value(vm[i]))
            end
            results[i][cost_fair][c_load_vary]["vm"] = voltage_magnitudes

            # save the line flows, current magnitudes and limits
            line_flow_percent = []
            line_names = []
            im_vec = []
            im_max_vec = []
            im_min_vec = []
            for (i,branch) in ref[:branch]
                f_idx = (i, branch["f_bus"], branch["t_bus"])
                push!(line_flow_percent,abs(value(p[f_idx]))/branch["rate_a"])
                push!(im_vec,branch["rate_a"]/abs(value(vm[branch["f_bus"]])-value(vm[branch["t_bus"]])))
                push!(im_min_vec,branch["rate_a"]/abs((ref[:bus][branch["f_bus"]]["vmin"] - ref[:bus][branch["t_bus"]]["vmin"])))
                push!(im_max_vec,branch["rate_a"]/abs((ref[:bus][branch["f_bus"]]["vmax"] - ref[:bus][branch["t_bus"]]["vmax"])))
                push!(line_names,"Line $i")
            end
            results[i][cost_fair][c_load_vary]["line_flow_percent"] = line_flow_percent
                    
            results[i][cost_fair][c_load_vary]["im"] = im_vec
            # Save the current limits 

            results[i][cost_fair][c_load_vary]["im_max"] = im_max_vec
            results[i][cost_fair][c_load_vary]["im_min"] = im_min_vec
            save("results_exp_$(i)_$(cost_fair).jld2", "results", results)
    end
end
# plot the percent load shed per bus for each test scenario and each value od the fairness cost
for i in 1:r
    flags = boolean_flags[i,:]
    true_flags = findall(flags)
    naming_test_files = ""
    for tf in true_flags
        naming_test_files = naming_test_files * "_" * scenarios[tf]
        #println("Naming test files: $naming_test_files")
    end
    for j in c_fair
# plot all the load shed percentages for each bus for all load
        k = c_load_vary
        ls_plt = plot()
            results = load("results_exp_$(i)_$(j).jld2", "results")
        # println("Results: ", keys(results_df_test))
            ls_percent_per_node = results[i][j][c_load_vary]["ls_percent"]
            ls_plt = scatter(net_buses, ls_percent_per_node, title="Load Shed Percentages", xlabel="Bus Number", ylabel="Load Shed Costs (USD/MW)", legend=true, label=@sprintf("%.2f",j))
            println("Total Load Served Percentages: ", sum(results[i][j][c_load_vary]["ls_percent"]))
        #end
        savefig(ls_plt, joinpath(save_folder,"percent_loadserved/percent_loadserved_$(naming_test_files)_fairness_$(j)_$network.png"))
# plot the voltage magnitudes for each bus for all load
        vm_plt = plot()
        for k in c_load_vary
            vm_plt = scatter!(net_buses, results[i][j][k]["vm"], title="Voltage Magnitudes", xlabel="Bus Number", ylabel="Voltage Magnitude (p.u.)", scatter=true, legend=true, label=@sprintf("%.2f",j))
        end
        savefig(vm_plt, joinpath(save_folder,"voltage_magnitudes/voltage_magnitudes_$(naming_test_files)_fairness_$(j)_$network.png"))
    
# plot the current magnitude and limits on the same figure
        im_plt = plot()
        for k in c_load_vary
            im_plt = scatter!(net_buses, results[i][j][k]["im"], title="Current Magnitudes", xlabel="Bus Number", ylabel="Current Magnitude (p.u.)", scatter=true, legend=true, label=@sprintf("%.2f",j))
            im_plt = scatter!(net_buses, results[i][j][k]["im_max"], title="Current Magnitudes", xlabel="Bus Number", ylabel="Current Magnitude (p.u.)",scatter=true,label=false)
            im_plt = scatter!(net_buses, results[i][j][k]["im_min"], title="Current Magnitudes", xlabel="Bus Number", ylabel="Current Magnitude (p.u.)",scatter=true,label=false)
        end
        savefig(im_plt, joinpath(save_folder,"current_magnitudes_and_limits/current_magnitudes_$(naming_test_files)_fairness_$(j)_$network.png"))
# plot the LMPs
        lmp_plt = plot()
        for k in c_load_vary
            lmp_plt = scatter!(net_buses, results[i][j][k]["lmp"]/(-ref[:baseMVA]), title="LMPs for Active Power Balance Constraints", xlabel="Bus Number", ylabel="LMP (USD/MW)", scatter=true, legend=true, label=@sprintf("%.2f",j))
        end
        savefig(lmp_plt, joinpath(save_folder,"lmp/lmp_$(naming_test_files)_fairness_$(j)_$network.png"))
    
    end

    # Plot a heatmap for the lmps, buses, and fairness cost
    lmp_heatmap = heatmap()
    heatmap_data = zeros(length(c_fair), length(net_buses))
    using PlotlyJS
    index = 0
    for j in c_fair
        results = load("results_exp_$(i)_$(j).jld2", "results")
        index+=1
            heatmap_data[index,:] = results[i][j][c_load_vary]["lmp"] / (-ref[:baseMVA])
            #println("Heatmap Data: ", heatmap_data)
            #println(size(heatmap_data))
    end
    lmp_heatmap = heatmap(heatmap_data, title="LMPs for Active Power Balance, Objective $(scenarios[i])", xlabel="Bus Number", ylabel="Fairness Cost", yticks=(1:length(c_fair), c_fair), color=:viridis)
    savefig(lmp_heatmap, joinpath(save_folder, "lmp/lmp_heatmap_$(naming_test_files)_$network.png"))

    # plot the percent loadshed per bus heatmap
    ls_heatmap = heatmap()
    ls_heatmap_data = zeros(length(c_fair), length(net_buses)-1)
    index = 0
    for j in c_fair
        results = load("results_exp_$(i)_$(j).jld2", "results")
        index+=1
        ls_heatmap_data[index,:] = results[i][j][c_load_vary]["ls_percent"]
    end
    ls_heatmap = heatmap(ls_heatmap_data, title="Load Served Percentages, Objective $(scenarios[i])", xlabel="Bus Number", ylabel="Fairness Cost",yticks=(1:length(c_fair), c_fair), color=:viridis)
    savefig(ls_heatmap, joinpath(save_folder, "percent_loadserved/percent_loadserved_heatmap_$(naming_test_files)_$network.png"))

    # Plot the Gini coefficient and Jain index
    # Assuming `results` is a nested dictionary like this:
    # results[objective_choice][fairness_cost][loadshed_cost] = Dict(:gini => gini_value, :jain => jain_value, ...)

    x_values = Float64[]  # Cost of load shedding (x-axis)
    y_values = Float64[]  # Cost of fairness (y-axis)
    z_gini_values = Float64[]  # Gini coefficient (z-axis)
    z_jain_values = Float64[]  # Jain index (z-axis)

    # Loop through the nested dictionary to extract the values
    for j in c_fair
        results = load("results_exp_$(i)_$(j).jld2", "results")
        gini_value = results[i][j][c_load_vary]["gini"]  # Extract the Gini coefficient
        jain_value = results[i][j][c_load_vary]["jain"]  # Extract the Jain index
        # Append the data to the respective arrays
        push!(y_values, j)
        push!(z_gini_values, gini_value)
        push!(z_jain_values, jain_value)
        println("Fairness Cost: $j, Gini Coefficient: $gini_value, Jain Index: $jain_value")
    end
    
    gini_plt = plot()
    gini_plt = scatter(y_values, z_gini_values, xlabel="Fairness Cost", ylabel="Gini Coefficient", 
            title="Gini Coefficient vs Fairness Cost, Objective $(scenarios[i])",
            marker=:circle, color=:blues, legend=false)

    jain_plt= plot()
    jain_plt = scatter(y_values, z_jain_values, xlabel="Fairness Cost", ylabel="Jain Index", 
            title="Jain Index vs Fairness Cost, Objective $(scenarios[i])",
            marker=:circle, color=:reds, legend=false)
    # Save the plot
    savefig(gini_plt, joinpath(save_folder, "gini/gini_$(naming_test_files)_$network.png"))
    savefig(jain_plt, joinpath(save_folder, "jain/jain_$(naming_test_files)_$network.png"))    
end

println("Total Percent Load Served: ", total_percent_load_served)

function lmp_calculation()
    #calculate the LMPs
    active_power_nodal_constraints = Dict(network_buses .=> model.ext[:constraints][:nodal_active_power_balance])
    lmp_vec = []
    bus_order = []
    for (i,bus) in ref[:bus]
        #divide the lmps by the baseMVA to get the lmps in $/MWh and then multiply by 1000 to get the lmps in $/kW
        push!(lmp_vec,dual(active_power_nodal_constraints[i])/(ref[:baseMVA]*10^3))
        push!(bus_order, bus["bus_i"])
    end
    return lmp_vec, bus_order, results
end

lmp_vec,bus_order = lmp_calculation()