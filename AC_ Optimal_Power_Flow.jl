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
using Dates 
using DataFrames

# Load System Data
# ----------------
powermodels_path = joinpath(dirname(pathof(PowerModels)), "../")
network = "case33bw_aw_edit"#"case5_matlab"#"case33bw"#"RTS_GMLC"#"case57"#"case118"#pglib_opf_case240_pserc"
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

#make a save folder for the plots 
if !isdir("plots_$date")
    mkdir("plots_$date")
end
save_folder = "plots_$date"
# Add zeros to turn linear objective functions into quadratic ones
# so that additional parameter checks are not required
PowerModels.standardize_cost_terms!(data, order=2)

# Adds reasonable rate_a values to branches without them
PowerModels.calc_thermal_limits!(data)

# use build_ref to filter out inactive components
ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
# note: ref contains all the relevant system parameters needed to build the OPF model
# When we introduce constraints and variable bounds below, we use the parameters in ref.

# Define the cost of load shedding
# Add the cost of the varying the loads (AW added 9/19/2024)
#c_load_vary = LinRange(2500, 6000, 50)
c_load_vary = LinRange(25, 20000, 2)
   # Assume the cost of varying the load is the same as the cost of generating power


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

# Add voltage angles vm for each bus
bus_vm = model.ext[:variables][:bus_vm] = @variable(model, ref[:bus][i]["vmin"] <= vm[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"], start=1.0)
# note: this vairable also includes the voltage magnitude limits and a starting value

# Add active power generation variable pg for each generator (including limits)
pg = model.ext[:variables][:pg] = @variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"])
# Add reactive power generation variable qg for each generator (including limits)
qg = model.ext[:variables][:qg] = @variable(model, ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= ref[:gen][i]["qmax"])

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
max_load = maximum([load["pd"] for (i,load) in ref[:load]])
min_load = 0.1*max_load #minimum([load["pd"] for (i,load) in ref[:load]])
p_load = model.ext[:variables][:p_load] = @variable(model, min_load <= p_load[i in keys(ref[:load])] <= max_load)

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
lmp_per_loadshed = Dict()
network_buses = keys(ref[:bus])
    net_buses = collect(network_buses)

        # Clean the plot
display(plot(1, 1, legend = false))

for cost_ls in c_load_vary
    # Minimize the cost of active power generation and cost of HVDC line usage
    # assumes costs are given as quadratic functions
    @objective(model, Min,
        sum(gen["cost"][1]*pg[i]^2 + gen["cost"][2]*pg[i] + gen["cost"][3] for (i,gen) in ref[:gen]) +
        sum(dcline["cost"][1]*p_dc[from_idx[i]]^2 + dcline["cost"][2]*p_dc[from_idx[i]] + dcline["cost"][3] for (i,dcline) in ref[:dcline]) +
        sum(cost_ls*(p_load[i]-load["pd"])^2 for (i,load) in ref[:load])
    )

    ###############################################################################
    # 3. Solve the Optimal Power Flow Model and Review the Results
    ###############################################################################

    # Solve the optimization problem
    optimize!(model)

    # Check that the solver terminated without an error
    println("The solver termination status is $(termination_status(model))")

    # Check the value of the objective function
    cost = objective_value(model)
    #println("The cost of generation is $(cost).")

    #sum the total load shed 
    for (i,load) in ref[:load]
        ls_amount = ref[:load][i]["pd"] - value(p_load[i])
    end
    ls_amount  =  sum(ls_amount)
    push!(ls_dict, [cost_ls, ls_amount])
    # save the load shedding cost and load shedding amount in a csv file with the cost_ls as the first column
    CSV.write(joinpath(save_folder,"load_shedding_$network.csv"),ls_dict)

    #AW updates 9/11/2024 and 9/20/2024 insert into the objective loop
    # Extract and store the LMPs for the active power balance constraints
    lmp = []
    
    active_power_nodal_constraints = Dict(network_buses .=> model.ext[:constraints][:nodal_active_power_balance])
    for (i,bus) in ref[:bus]
        push!(lmp,dual(active_power_nodal_constraints[i]))
    end
    lmp_dict = Dict(network_buses .=> lmp)
    lmp_per_loadshed[cost_ls] = lmp_dict
    # Print the LMPs for the active power balance constraints
    # println("The LMPs for the active power balance constraints are:")
    # for (i,lmp_i) in lmp_dict
    #     println("LMP at bus $i is $(lmp_i).")
    # end

    #plot the LMPs
    sct = scatter!(net_buses, lmp/-ref[:baseMVA], title="LMPs for Active Power Balance Constraints", xlabel="Bus Number", ylabel="LMP (USD/$units h)", scatter=true, legend=true, label="$cost_ls")
    # note: the LMPs are in $/MWh
    # save the Plot
    # Store the LMPs in a csv file
    #CSV.write("lmp.csv", lmp)
    savefig(sct,joinpath(save_folder,"lmp_$network.png"))
end
    # Clean the plot
display(plot(1, 1, legend = false))

# Plot the load shed cost (y-axis) vs. load shed (x-axis)
ls_costs = ls_dict[!,"ls_cost"] 
ls_amounts = ls_dict[!,"ls_amount"]
plt = scatter(ls_costs,ls_amounts, title = "Load Shed Costs vs. Load Shed Amount",  xlabel = "Load Shed Cost (USD/MW p.u.)", ylabel = "Cummulative Load Shed Amount (MW p.u.)", legend = false)
savefig(plt, joinpath(save_folder,"Loadshed_and_Cost_$network.png"))
# Check the value of an optimization variable
# Example: Active power generated at generator 1
pg1 = value(pg[1])
println("The active power generated at generator 1 is $(pg1*ref[:baseMVA]) $units.")
# note: the optimization model is in per unit, so the baseMVA value is used to restore the physical units

#print the total active load
println("The total active load is $(sum([value(p_load[i]) for (i,load) in ref[:load]])*ref[:baseMVA]) $units.")
#######

# Plot the voltage angles
voltage_angles = []
for (i,bus) in ref[:bus]
    push!(voltage_angles,value(va[i]))
end
scatter(net_buses, voltage_angles, title="Voltage Angles", xlabel="Bus Number", ylabel="Voltage Angle (rad)", scatter=true, legend=false)
# note: the voltage angles are in radians
# save the Plot
savefig(joinpath(save_folder,"voltage_angles_$network.png"))

# Plot the voltage magnitudes
voltage_magnitudes = []
for (i,bus) in ref[:bus]
    push!(voltage_magnitudes,value(vm[i]))
end
scatter(net_buses, voltage_magnitudes, title="Voltage Magnitudes", xlabel="Bus Number", ylabel="Voltage Magnitude (p.u.)", scatter=true, legend=false)
# note: the voltage magnitudes are in per unit
# save the Plot
savefig("voltage_magnitudes_$date.png")
# Plot the voltage magnitudes and limits on the same Plot
voltage_max_limits = []
voltage_min_limits = []
for (i,bus) in ref[:bus]
    push!(voltage_max_limits,ref[:bus][i]["vmax"])
    push!(voltage_min_limits,ref[:bus][i]["vmin"])
end
scatter(net_buses, voltage_magnitudes, title="Voltage Magnitudes and Limits", xlabel="Bus Number", ylabel="Voltage Magnitude (p.u.)", label="voltage_magnitudes", legend=false)
plot!(net_buses, voltage_max_limits, line=(:dash), title="Voltage Magnitudes and Limits", xlabel="Bus Number", ylabel="Voltage Magnitude (p.u.)", label="voltage_magnitudes_max_limits", legend=false)
plot!(net_buses, voltage_min_limits, line=(:dash), title="Voltage Magnitudes and Limits", xlabel="Bus Number", ylabel="Voltage Magnitude (p.u.)", label="voltage_magnitudes_min_limits", legend=true)
# note: the voltage magnitudes and limits are in per unit
# save the Plot
savefig(joinpath(save_folder,"voltage_magnitudes_and_limits_$network.png"))

# Plot the line flows
line_flows = []
line_names = []
for (i,branch) in ref[:branch]
    f_idx = (i, branch["f_bus"], branch["t_bus"])
    push!(line_flows,abs(value(p[f_idx])))
    push!(line_names,"Line $i")
end
bar(line_names, line_flows, title="Line Flow Magnitudes", xlabel="Line Number", ylabel="Active Power Flow ($units p.u.)", legend=false)
# note: the line flows are in per unit
# save the Plot
savefig(joinpath(save_folder,"line_flows_$network.png"))

# PLot the line flows and line limits on the same Plot
line_limits = []
for (i,branch) in ref[:branch]
    push!(line_limits,branch["rate_a"])
end
bar(line_names, line_flows, title="Line Flows and Limits", xlabel="Line Number", ylabel="Active Power Flow ($units p.u.)",label = "line_flows", legend=false)
plot!(line_names, line_limits, line=(:dash), title="Line Flows and Limits", xlabel="Line Number", ylabel="Active Power Flow ($units p.u.)", label = "line limits", legend=true)
# note: the line flows and limits are in per unit
# save the Plot
savefig(joinpath(save_folder,"line_flows_and_limits_$network.png"))

#powerplot(data)

######## 
#AW updates 9/13/2024
# DCOPF version for sanity
# Run DC power flow formulation
# Get duals and solution to dcopf
# sol = solve_dc_opf(file_name, Gurobi.Optimizer, setting = Dict("output"=>Dict("duals"=>true)))
# @assert sol["solution"]["per_unit"] == true
# @assert sol["termination_status"] == MOI.OPTIMAL
# # Extract the LMPs for the active power balance constraints
# λ_kcl_opt = [b["lam_kcl_r"] for (i,b) in sol["solution"]["bus"]]

# #For $/MWh, baseMVA = 100
# lmp  = λ_kcl_opt/(-100)
