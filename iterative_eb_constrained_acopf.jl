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
using Statistics
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

# Define the cost of load shedding
c_load_vary = 2e3#LinRange(200, 2000, 2), 200 same as gen cost so shed load, 2000 too high no load shed
# Assume the cost of varying the load is the same as the cost of generaxting power

# Define the cost of fairness in load shedding 
c_fair = LinRange(1, 1E2, 2)#*1e2

# Define a parameter to scale the load
# Increasing will create an undervoltage problem
scale_load = 1.0
scale_gen = scale_load

# Define the threshold for the energy burden, aka the energy poverty threshold
eb_threshold = 0.06 # 6% of the average load
#dlmp_base = ones(length(ref[:load]))#rand(0.5:0.01:1.5, length(data["load"]))*.2 #.2 $/kWh
# Add zeros to turn linear objective functions into quadratic ones
# so that additional parameter checks are not required
PowerModels.silence()

function acopf_main(eb_constraint_flag, pv_constraint_flag, pv_gen_max, data, scale_load, scale_gen, eb_threshold, dlmp_base, c_load_vary, c_fair, date, save_folder, iteration_number)
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
    # set the power models output to nothing
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

    if eb_constraint_flag
        # Add the energy burden constraint
        #model.ext[:variables][:energy_burdens] = @variable(model, eb[i in keys(ref[:load])]>=0)
        # Additional parameter to ensure the dlmp input into the eb constraint is non-negative
        dlmp_positive = -dlmp_base 
        eb = ([load["pd"]*ref[:baseMVA] for (i,load) in ref[:load]].*dlmp_positive)./(3000000/(365*24))
        #@expression(model, eb .= ([load["pd"] for (i,load) in ref[:load]].*dlmp_base)./30000)
        model.ext[:constraints][:energy_burden] = @constraint(model, eb .<= eb_threshold)
        println("Length of energy burden vector: ", length(eb))
        println("Length of load vector: ", length([load["pd"] for (i,load) in ref[:load]]))
        println("Length of dlmp_base vector: ", length(dlmp_base))
    end 

    if pv_constraint_flag
        # Add the PV generation constraint
        dlmp_positive = -dlmp_base 
        @variable(model, pv_gen[i in keys(ref[:load])] >= 0)
        @constraint(model, pv_gen[i in keys(ref[:load])] <= pv_gen_max)
        @constraint(model, sum(([load["pd"]*ref[:baseMVA] for (i,load) in ref[:load]]-[pv_gen[i]*ref[:baseMVA] for (i,load) in ref[:load]]).*dlmp_positive)/70000 <= eb_threshold)
        # for bus in keys(ref[:bus])
        #     if ref[:bus][bus]["type"] == 2
        #         pv_size = 0.1
        #         pv_gen_match = ref[:bus][bus]["pd"]*pv_size
        #         ref[:bus][bus]["pd"] = ref[:bus][bus]["pd"] - pv_gen_match
        #         ref[:gen][bus]["pg"] = ref[:gen][bus]["pg"] + pv_gen_match
        #     end
        # end
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
        #for cost_ls in c_load_vary

            # Minimize the cost of active power generation and cost of HVDC line usage
            # assumes costs are given as quadratic functions
            @objective(model, Min, acopf_obj
                            + load_shed_obj)

            ###############################################################################
            # 3. Solve the Optimal Power Flow Model and Review the Results
            ###############################################################################

            # Solve the optimization problem
            optimize!(model)

            # Check that the solver terminated without an error
            println("The solver termination status is $(termination_status(model))")
            network_buses = keys(ref[:bus])

            # save the active power lmps
            active_power_nodal_constraints = Dict(network_buses .=> model.ext[:constraints][:nodal_active_power_balance])
            lmp_vec_pf = []
            bus_order = []
            eb_out = []
            loads = []
            for (i,bus) in ref[:bus]
                if i != 1
                    push!(bus_order, i)
                    push!(lmp_vec_pf,dual(active_power_nodal_constraints[i]))
                end
            end
            save("lmp_.jld2", "lmp", lmp_vec_pf/(-ref[:baseMVA]))
            if eb_constraint_flag
                for (i,load) in ref[:load]
                    if iteration_number == 1
                        push!(eb_out, 1)
                    else 
                        push!(eb_out, value(eb[i]))
                    end
                    push!(loads, load["pd"]*ref[:baseMVA])
                end
            end
            basemva = ref[:baseMVA]
    return lmp_vec_pf, bus_order,eb_out, loads, basemva
end

# save the iteration number and the mean of the lmps
# Initialize variables
tol_check = []
lmp_mean = []
lmp_results = []
tol = 100.0  # Set initial tolerance value to a high number
global dlmp_base = ones(length(data["load"]))*.5
#global lmp_vec = dlmp_base
# Loop until the maximum difference between iterations is below the tolerance
# for iteration in 1:3
#     #maximum(abs.(tol)) >= 0.001
#     println("Iteration: ", iteration)
#     # Call acopf_main based on the iteration count to initialize or update lmp_vec
#     if iteration == 1
#         eb_constraint_flag = false
#         global lmp_vec = zeros(length(data["load"]))
#     else
#         eb_constraint_flag = true
#     end
#     # Calculate the tolerance between iterations and update if above threshold
#     dlmp,bus_order,eb, loads, basemva = acopf_main(eb_constraint_flag, data, scale_load, scale_gen, eb_threshold, lmp_vec, c_load_vary, c_fair, date, save_folder, iteration)

#     #     global lmp_vec = acopf_main(eb_constraint_flag, data, scale_load, scale_gen, eb_threshold, dlmp, c_load_vary, c_fair, date, save_folder)
#     # end
#         #println("The LMP_vec: ", lmp_vec)
#     tolerance_check = abs.(lmp_vec) .- abs.(dlmp)
#     tol = abs.(tolerance_check)
#     push!(lmp_mean, mean(lmp_vec))
#     push!(tol_check, tolerance_check)
#     push!(lmp_results, lmp_vec)

#     # Update dlmp for the next iteration
#     #println("Current DLMPs: ", dlmp)
#     #println("The LMP_vec: ", lmp_vec)
#     #println("Tolerance Check: ", tolerance_check)
#     lmp_vec = dlmp
#     #println("Tolerance: ", tol_check)
#     println("Energy Burden: ", eb)
#     #println("Loads: ", loads)
#     #println("Base Power: ", basemva)
#     # Increment iteration counter
#     # if maximum(abs.(tol)) .< 0.01
#     #     break
#     # end
# end

#run for 7  days in 15 minute intervals for base case with new load data
# one day is 1440 minutes
# 7 days is 10080 minutes
# 15 minute intervals is 96 intervals per day
# 96*7 = 672 intervals
num_days = 2#7
time_intervals = 15 #minutes
int_per_day = 96 #intervals per day
eb_constraint_flag = false
pv_constraint_flag = false
lmp_results_daily_base = DataFrame(
    Day = Int[],
    Bus = Int[],
    LMP = Float64[],
)
lmp_results_minutes_base = DataFrame(
    Interval = Int[],
    Bus = Int[],
    LMP = Float64[],
)
lmp_results_daily_pv_gen = DataFrame(
    Day = Int[],
    Bus = Int[],
    LMP = Float64[],
)
lmp_vec = zeros(length(data["load"]))
net_buses = keys(data["bus"])
for d in 1:num_days
    for iteration in 1:int_per_day
        #run the opf with the load data associated with this time interval and this day
        for loads in keys(data["load"])
            #println("Load: ", data["load"][loads]["pd"])
            data["load"][loads]["pd"] = data["load"][loads]["pd"]#data["load"][d][iteration]
        end
        # Call acopf_main based on the iteration count to initialize or update lmp_vec
        dlmp,bus_order,eb, loads, basemva = acopf_main(eb_constraint_flag, pv_constraint_flag, data, scale_load, scale_gen, eb_threshold, lmp_vec, c_load_vary, c_fair, date, save_folder, iteration)
        for bus in keys(dlmp)
            push!(lmp_results_minutes_base, (d, bus, dlmp[bus]))
        end
    end 
    dlmp,bus_order,eb, loads, basemva = acopf_main(eb_constraint_flag, pv_constraint_flag, data, scale_load, scale_gen, eb_threshold, lmp_vec, c_load_vary, c_fair, date, save_folder, 1)
    for bus_id in keys(dlmp)
        lmp_results_filtered = filter(row -> row.Bus == bus_id, lmp_results_minutes_base)
        #println("LMP Results Filtered: ", lmp_results_filtered)
        result = sum(lmp_results_filtered[:,:LMP])/length(lmp_results_filtered[:,:LMP])
        # Display the result
        push!(lmp_results_daily_base, (d, bus_id, result))
    end
end 

#create a heatmap of the lmps for each bus over the 7 days
# Plot a heatmap for the lmps, buses, and fairness cost
lmp_heatmap = heatmap()
heatmap_data = zeros(num_days, length(net_buses))
using PlotlyJS
index = 0
for i in num_days
    for j in net_buses
        lmp_results_filtered = filter(row -> row.Day == i, lmp_results_daily_base)
        #index += 1
        println("LMP Results Filtered: ", lmp_results_filtered[:,3])
        heatmap_data[i,:] .= lmp_results_filtered[:,3][1] ./ (-ref[:baseMVA])
        #println("Heatmap Data: ", heatmap_data)
        #println(size(heatmap_data))
    end
end
lmp_heatmap = heatmap(heatmap_data, title="LMPs for Active Power Balance, Objective", xlabel="Bus Number", ylabel="Day", color=:viridis)
savefig(lmp_heatmap, joinpath(save_folder, "lmp/lmp_heatmap.png"))


# insert the PV generation constraint into the problem and run again
#Re-run the opf with the new load data including the pv generation
pv_constraint_flag = true
for d in 1:num_days
    for iteration in 1:int_per_day
        # time_steps = (d-1)*int_per_day + iteration
        #run the opf with the load data associated with this time interval and this day
        for loads in keys(data["load"])
            #println("Load: ", data["load"][loads]["pd"])
            data["load"][loads]["pd"] = data["load"][loads]["pd"]#-pv_gen_data[loads][time_steps]
        end
        #pv_gen_max = pv_gen_data[loads][time_steps]
        # Call acopf_main based on the iteration count to initialize or update lmp_vec
        dlmp,bus_order,eb, loads, basemva = acopf_main(eb_constraint_flag, pv_constraint_flag, pv_gen_max, data, scale_load, scale_gen, eb_threshold, lmp_vec, c_load_vary, c_fair, date, save_folder, iteration)
        for bus in keys(dlmp)
            push!(lmp_results_daily_pv_gen, (d, bus, dlmp[bus]))
        end
    end 
end 
#println("Convergence achieved after ", iterations, " iterations.")

# Plot a heatmap for the lmps, buses, and fairness cost; for the PV generation case
lmp_pv_heatmap = heatmap()
heatmap_data = zeros(num_days, length(net_buses))
using PlotlyJS
index = 0
for i in num_days
    for j in net_buses
        lmp_results_filtered = filter(row -> row.Day == i, lmp_results_daily_pv_gen)
        #index += 1
        println("LMP Results Filtered: ", lmp_results_filtered[:,3])
        heatmap_data[i,:] .= lmp_results_filtered[:,3][1] ./ (-ref[:baseMVA])
        #println("Heatmap Data: ", heatmap_data)
        #println(size(heatmap_data))
    end
end
lmp_pv_heatmap = heatmap(heatmap_data, title="LMPs for Active Power Balance, Objective", xlabel="Bus Number", ylabel="Day", color=:viridis)
savefig(lmp_pv_heatmap, joinpath(save_folder, "lmp/lmp_heatmap.png"))