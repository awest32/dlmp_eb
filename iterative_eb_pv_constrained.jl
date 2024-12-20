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
#include("../index.jl")
include("pv_data/data_inputs.jl")
# Load System Data
# ----------------
powermodels_path = joinpath(dirname(pathof(PowerModels)), "../")
network = "case33bw_aw_edit" #"case5_matlab"#"case33bw"#"RTS_GMLC"#"case57"#"case118"#pglib_opf_case240_pserc"
units = "MW"

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
# Assume the cost of varying the load is the same as the cost of generaxting power



# Define the threshold for the energy burden, aka the energy poverty threshold
eb_threshold = 0.06 # 6% of the average load

PowerModels.silence()

function acopf_main(data, days,time_steps, scale_load, scale_gen, pv_constraint_flag, date, save_folder, iteration_number)
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

    # Create the p_load parameter for each load, day, and timestep
    p_load = model.ext[:variables][:p_load] = Dict()
    p_load = model.ext[:variables][:p_load] = @variable(model, p_load[i in keys(ref[:load]), 1:days, 1:time_steps])

    for i in keys(ref[:load])
        #println("Load: ", i)
        for d in 1:days
         #   println("Day: ", d)
            for t in 1:time_steps
          #     println("Time Step: ", t)
                #println("scale_load: ", scale_load[scale_load.day .== d, 2][t])
                @constraint(model,p_load[i,d,t] .== scale_load[scale_load.day .== d, 2][t] * ref[:load][i]["pd"])
            end
        end
    end

    # Add a variable to represent the cost of active pv power at each bus 
    x_pv = model.ext[:variables][:x_pv] = @variable(model, 0 <= x_pv[i in keys(ref[:load]), 1:days, 1:time_steps])
    for i in keys(ref[:load])
        for d in 1:days
            for t in 1:time_steps
                if pv_constraint_flag == true
                    @constraint(model,x_pv[i,d,t] .<= scale_load[scale_load.day .== d, 2][t] * ref[:load][i]["pd"])
                else
                    JuMP.set_upper_bound(x_pv[i,d,t], 0)#scale_load[scale_load.day .== d, 2][t] * ref[:load][i]["pd"])
                end
            end
        end
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

        # include the pv in the power balance as positive power
        push!(model.ext[:constraints][:nodal_active_power_balance],@constraint(model,
            sum(p[a] for a in ref[:bus_arcs][i]) +                  # sum of active power flow on lines from bus i +
            sum(p_dc[a_dc] for a_dc in ref[:bus_arcs_dc][i]) ==     # sum of active power flow on HVDC lines from bus i =
            sum(pg[g] for g in ref[:bus_gens][i]) -                 # sum of active power generation at bus i -
            sum(p_load[l,d,t] for l in ref[:bus_loads][i] for d in 1:days for t in 1:time_steps) +        # sum of active load consumption at bus i -
            sum(x_pv[pv,d,t] for pv in ref[:bus_loads][i] for d in 1:days for t in 1:time_steps) -  # sum of PV generation at bus i +
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

   
            @objective(model, Min, acopf_obj)

        return model, ref[:load], ref[:baseMVA], ref[:bus]
end
function run_acopf(model, ref_load, ref_baseMVA, ref_bus, scale_load, scale_pv, eb_threshold, pv_constraint_flag)
    # Run the ACOPF model
    # -------------------
    # Define the solver
    set_optimizer(model, Ipopt.Optimizer)
    # Set the solver options
    set_optimizer_attributes(model, "print_level" => 0)
    # Solve the optimization problem
    optimize!(model)
    # Extract the results
    # -------------------
    # Extract the lmps
    network_buses = keys(ref_bus)
    active_power_nodal_constraints = Dict(network_buses .=> model.ext[:constraints][:nodal_active_power_balance])
    lmp_vec_pf = []
    bus_order = []
    eb_out = []
    pv_out = []
    loads = []
    
    # Initialize DataFrame with timesteps
    lmp_df = DataFrame(bus=Int[], day=Int[], timestep=Int[], lmp=Float64[])
    
    # Extract LMPs for each bus and timestep
    for (i,bus) in ref_bus
            push!(bus_order, bus["bus_i"])
            # Get LMP for this bus
            lmp_value = dual(active_power_nodal_constraints[i])
            push!(lmp_vec_pf, lmp_value)
            
            # Add LMP values for each timestep
            for d in 1:days
                for t in 1:time_steps
                    #Divide by the baseMVA to convert the lmps into $/MWh, divide by -1 to be positive
                    push!(lmp_df, [bus["bus_i"], d, t, lmp_value/(-ref_baseMVA)])
                end
            end
        end

    #export the value of the p_load and ref_load*scale_load at each time step per day and bus
    ref_load_scaled_out = DataFrame(bus=Int[], day=Int[], load=Float64[])
    
    p_load_out = value.(model.ext[:variables][:p_load])
    for i in keys(ref_load)
        for d in 1:days
            for t in 1:time_steps
            push!(ref_load_scaled_out, [i, d, ref_load[i]["pd"]*scale_load[scale_load.day .== d, 2][t]])
            end
        end
    end
    #save("lmp_.jld2", "lmp", lmp_vec_pf/(-ref_baseMVA))
    return lmp_vec_pf, lmp_df, bus_order, ref_load_scaled_out, p_load_out
end
eb_constraint_flag = false
pv_constraint_flag = false
#set the lower and upper bound of the x_pv variables to zero
date = Dates.today()
iteration_number = 1
current_dir = pwd()
load_data_name = current_dir*"/pv_data/Pload_p.csv"
pv_data_name = current_dir*"/pv_data/pv_gen.csv"
ghi_data_name = current_dir*"/pv_data/ghi.csv"
load_data_normed, load_data_raw = read_load_data(load_data_name)
pv_data_normed, ghi_data_normed = read_pv_data(pv_data_name, ghi_data_name)
scale_pv = ghi_data_normed
scale_gen = 1
time_steps = 288
days = 8
pv_lcoe = 200

extr_load, extr_pv = extract_small_amount_of_time(7,4, 2, load_data_normed, scale_pv, 5)
extr_load_more, extr_pv_more = extract_small_amount_of_time(4,4, 2, load_data_normed, scale_pv, 5)
extr_pv[extr_pv.day .== 7, 2] = extr_pv_more[extr_pv_more.day .==7, 2]
scale_load = extr_load
scale_pv = extr_pv

# Iteration Zero: No PV Energy Burden Constraints
acopf_model, ref_load, ref_baseMVA, ref_bus = acopf_main(data, days, time_steps, scale_load, scale_gen, pv_constraint_flag, date, save_folder, iteration_number)
dlmp_base,lmp_df, bus_order, ref_load_scaled_out, p_load_returned = run_acopf(acopf_model, ref_load, ref_baseMVA, ref_bus, scale_load, scale_pv, eb_threshold, pv_constraint_flag)

generation_costs_iter_zero = objective_value(acopf_model)


pv_constraint_flag = true
eb_constraint_flag = false

#acopf_model, ref_load, ref_baseMVA, ref_bus = acopf_main(data, days, time_steps, scale_load, scale_gen, pv_constraint_flag, date, save_folder, iteration_number)

# Add the PV energy burden generation constraints for each day and timestep
function add_eb_constraints(acopf_model, lmp_df, ref_load, ref_baseMVA, ref_bus, scale_load, scale_pv, eb_threshold, pv_constraint_flag)
    acopf_model.ext[:constraints][:eb] = Dict()
    for day in 1:days
        for tstep in 1:time_steps
            lmps = []
            for i in keys(ref_load)
                push!(lmps, lmp_df.lmp[(lmp_df.day .== day) .& (lmp_df.bus .== i) .& (lmp_df.timestep .== tstep)][1])
            end
            #println("The LMPs for day $day and timestep $tstep are: $lmps")
            normal_loads = [extr_load[extr_load.day .== day, 2][tstep]*load["pd"]*ref_baseMVA for (i,load) in ref_load]
            pv_amounts = [extr_pv[extr_pv.day .== day,2][tstep]*acopf_model.ext[:variables][:x_pv][i,day,tstep]*ref_baseMVA for (i,load) in ref_load]
            load_diff = normal_loads - pv_amounts
            #eb_frac = load_diff/lmps
            acopf_model.ext[:constraints][:eb][day,tstep] = @constraint(acopf_model, sum((normal_loads.-pv_amounts).*lmps)./70000 <= eb_threshold)
        end
    end
    return acopf_model
end

#acopf_model = add_eb_constraints(acopf_model, lmp_df, ref_load, ref_baseMVA, ref_bus, scale_load, scale_pv, eb_threshold, pv_constraint_flag)

function add_pv_costs_to_objective(acopf_model, pv_lcoe)
    # Calculate the cost of PV generation
    @expression(acopf_model, pv_cost_obj, pv_lcoe * sum(acopf_model.ext[:variables][:x_pv][i,d,t] for i in keys(ref_load) for d in 1:days for t in 1:time_steps))

    # Get the existing objective expression
    existing_obj = objective_function(acopf_model)

    # Update the objective to include both the original objective and PV cost
    @objective(acopf_model, Min, existing_obj + pv_cost_obj)
end

# Add the PV cost to the objective function
#add_pv_costs_to_objective(acopf_model, pv_lcoe)

# Main iteration loop
max_iterations = 10
convergence_threshold = 0.001  # 0.1% difference between iterations
generation_costs_per_iteration = DataFrame(
    Iteration = Int[],
    Generation_Cost = Float64[]
)

lmp_df_list = []
push!(lmp_df_list, lmp_df)
iteration = 0
push!(generation_costs_per_iteration, [iteration, generation_costs_iter_zero])

# Add DataFrames to store metrics per iteration
metrics = DataFrame(
    iteration = Int[],
    cost = Float64[],
    total_load = Float64[],
    total_pv = Float64[],
    mean_burden = Float64[],
    max_burden = Float64[]
)
# Calculate metrics for this iteration
    # Get the cost for timestep t
    cost = value(objective_value(acopf_model))
    # Get the total load served for timestep t
    tot_load = []
    for i in keys(ref_load)
        for d in 1:days
            for t in 1:time_steps
               push!(tot_load,value(acopf_model.ext[:variables][:p_load][i,d,t]))
            end
        end
    end
    total_load_served = sum(tot_load)
    # Get the total PV generation for timestep t
    tot_pv = []
    total_pv_generation_per_bus = []
    for i in keys(ref_load)
        tot_pv_bus = []
        for d in 1:days
            for t in 1:time_steps
               push!(tot_pv_bus, value(acopf_model.ext[:variables][:x_pv][i,d,t]))
            end
            push!(tot_pv, tot_pv_bus)
        end
        push!(total_pv_generation_per_bus, sum(tot_pv_bus))
    end
    total_pv_generation = sum(total_pv_generation_per_bus)
    # Get the total load for timestep t
    tot_load = []
    total_load_per_bus = []
    for i in keys(ref_load)
        tot_load_per_bus = []
        for d in 1:days
            for t in 1:time_steps
               push!(tot_load_per_bus,ref_load[i]["pd"]*scale_load[scale_load.day .== d, 2][t])
            end
        end
        push!(total_load_per_bus, sum(tot_load_per_bus))
        push!(tot_load, sum(tot_load_per_bus))
    end
    total_load = sum(tot_load)
    # Calculate the burden for timestep t, should be a vector of all of the burden per bus
    burden = (total_load_per_bus .- total_pv_generation_per_bus)./total_load_per_bus
    # Calculate the max burden for timestep t
    max_burden = maximum(burden)
    # Calculate the mean burden for timestep t
    mean_burden = mean(burden)
    push!(metrics, [iteration, cost, total_load, total_pv_generation, mean_burden, max_burden])
    
for iteration in 1:max_iterations
    current_dir = pwd()
    load_data_name = current_dir*"/pv_data/Pload_p.csv"
    pv_data_name = current_dir*"/pv_data/pv_gen.csv"
    ghi_data_name = current_dir*"/pv_data/ghi.csv"
    load_data_normed, load_data_raw = read_load_data(load_data_name)
    pv_data_normed, ghi_data_normed = read_pv_data(pv_data_name, ghi_data_name)
    scale_pv = ghi_data_normed
    data = PowerModels.parse_file("case33bw_aw_edit.m")
    # 7 days offset from max days, 4 seasons, 2 days per season, 5 minutes per timestep
    extr_load, extr_pv = extract_small_amount_of_time(7, 4, 2, load_data_normed, scale_pv, 5)
    extr_load_more, extr_pv_more = extract_small_amount_of_time(4, 4, 2, load_data_normed, scale_pv, 5)
    extr_pv[extr_pv.day .== 7, 2] = extr_pv_more[extr_pv_more.day .==7, 2]
    scale_load = extr_load
    scale_pv = extr_pv

    days = 8 # The number of days to simulate
    time_steps = 288 # The number of time steps per day
    scale_gen = 1 
    pv_constraint_flag = true
    date = Dates.today() 
    save_folder = "plots_$date/"
    scale_load = extr_load 

    # Run the ACOPF model
    acopf_model, ref_load, ref_baseMVA, ref_bus = acopf_main(data, days, time_steps, scale_load, scale_gen, pv_constraint_flag, date, save_folder, iteration)
    # Add the PV energy burden generation costs for each day and timestep
    add_pv_costs_to_objective(acopf_model, pv_lcoe)
    # Add the PV energy burden generation constraints for each day and timestep
    old_lmp_df = lmp_df_list[end]
    acopf_model = add_eb_constraints(acopf_model, old_lmp_df, ref_load, ref_baseMVA, ref_bus, scale_load, scale_pv, eb_threshold, pv_constraint_flag)
    
    # Run ACOPF and get LMPs
    lmp_vec_pv, lmp_df, bus_order, ref_scaled_out, p_load = run_acopf(acopf_model, ref_load, ref_baseMVA, ref_bus, scale_load, scale_pv, eb_threshold, pv_constraint_flag)
    push!(lmp_df_list, lmp_df)  

    # Calculate metrics for this iteration
    # Get the cost for timestep t
    cost = value(objective_value(acopf_model))
    # Get the total load served for timestep t
    tot_load = []
    for i in keys(ref_load)
        for d in 1:days
            for t in 1:time_steps
               push!(tot_load,value(acopf_model.ext[:variables][:p_load][i,d,t]))
            end
        end
    end
    total_load_served = sum(tot_load)
    # Get the total PV generation for timestep t
    tot_pv = []
    total_pv_generation_per_bus = []
    for i in keys(ref_load)
        tot_pv_bus = []
        for d in 1:days
            for t in 1:time_steps
               push!(tot_pv_bus, value(acopf_model.ext[:variables][:x_pv][i,d,t]))
            end
            push!(tot_pv, tot_pv_bus)
        end
        push!(total_pv_generation_per_bus, sum(tot_pv_bus))
    end
    total_pv_generation = sum(total_pv_generation_per_bus)
    # Get the total load for timestep t
    tot_load = []
    total_load_per_bus = []
    for i in keys(ref_load)
        tot_load_per_bus = []
        for d in 1:days
            for t in 1:time_steps
               push!(tot_load_per_bus,ref_load[i]["pd"]*scale_load[scale_load.day .== d, 2][t])
            end
        end
        push!(total_load_per_bus, sum(tot_load_per_bus))
        push!(tot_load, sum(tot_load_per_bus))
    end
    total_load = sum(tot_load)
    # Calculate the burden for timestep t, should be a vector of all of the burden per bus
    burden = (total_load_per_bus .- total_pv_generation_per_bus)./total_load_per_bus
    # Calculate the max burden for timestep t
    max_burden = maximum(burden)
    # Calculate the mean burden for timestep t
    mean_burden = mean(burden)
    # Add the metrics for timestep t to the metrics dataframe
    push!(metrics, [iteration, cost, total_load, total_pv_generation, mean_burden, max_burden])
    

    # Calculate current generation cost
    current_cost = objective_value(acopf_model)
    push!(generation_costs_per_iteration, [iteration, current_cost])
    println("Generation cost for iteration $iteration: $(generation_costs_per_iteration[end,2])")
    
    println("Generation cost for iteration $iteration: $(generation_costs_per_iteration[end,2])")
    previous_cost = generation_costs_per_iteration[end-1,2]
    println("Previous generation cost for iteration $(iteration): $(generation_costs_per_iteration[end-1,2])")
    #Check for convergence
    if abs((current_cost - previous_cost)) < convergence_threshold && iteration > 1
        println("Converged after $iteration iterations!")
        break
    end
    
    # Plot LMPs for this iteration
    lmp_plot = plot()
    for bus in unique(lmp_df.bus)
        bus_data = lmp_df[(lmp_df.bus .== bus) .& (lmp_df.day .== 2), :]
        plot!(lmp_plot, bus_data.timestep, bus_data.lmp, 
              label="Bus $bus", 
              linewidth=2, 
              marker=:circle,
              markersize=3)
    end
    xlabel!(lmp_plot, "Time Step")
    ylabel!(lmp_plot, "LMP (\$/MWh)")
    title!(lmp_plot, "LMPs per Bus - Day 2, Iteration $iteration")
    savefig(lmp_plot, "plots_$date/lmp_profiles_iteration_$iteration.png")
end

# Plot convergence
convergence_plot = plot(1:length(generation_costs_per_iteration.Iteration), generation_costs_per_iteration[:,2],
                       xlabel="Iteration",
                       ylabel="Generation Cost",
                       title="Generation Cost Convergence",
                       marker=:circle,
                       label="Generation Cost")
savefig(convergence_plot, "plots_$date/convergence.png")

println("Final generation costs across iterations: ", generation_costs_per_iteration[:,2])

# dlmp_base,lmp_df, bus_order,p_load = run_acopf(acopf_model, ref_load, ref_baseMVA, ref_bus, scale_load, scale_pv, eb_threshold, pv_constraint_flag)



# # Plot for a specific bus (e.g., bus 5)
# bus_to_plot = 5
# for d in 1:days
#     bus_data = filter(row -> row.bus == bus_to_plot && row.day == d, load_per_bus)
#     plot!(load_plot, bus_data.timestep, bus_data.load, 
#           xlabel="Time Step", 
#           ylabel="Load (MW)", 
#           title="Demand for Bus $bus_to_plot Across Eight Representative Days", 
#           label="Day $d",
#           linewidth=2,
#           marker=:circle,
#           markersize=3)
# end
# savefig(load_plot, "plots_$date/load_day.png")

# lmp_vec_pv, lmp_df, bus_order,ref_scaled_out, p_load = run_acopf(acopf_model, ref_load, ref_baseMVA, ref_bus, scale_load, scale_pv, eb_threshold, pv_constraint_flag)
# #caculate the energy burden per bus per day and time step
# eb = DataFrame(bus=Int[], day=Int[], tstep=Int[], eb=Float64[])
# for (i,b) in ref_load
#     for d in 1:days
#         for t in 1:time_steps
#             dlmp_positive = -lmp_df[lmp_df.bus .== b["index"],2]
#             if b["index"] == 1
#                 dlmp_positive = 0
#             end
#             eb_day = (value(acopf_model.ext[:variables][:p_load][i][d][t])-value(acopf_model.ext[:variables][:x_pv][b["index"],d,t]))*dlmp_positive/70000
#             eb_day = eb_day[1]
#             push!(eb,[b["index"],d,t,eb_day])
#         end
#     end
# end

# # After optimization, collect metrics
# cost = objective_value(acopf_model)
# total_pv = sum(value.(acopf_model.ext[:variables][:x_pv]))
# mean_burden = mean(eb.eb)
# max_burden = maximum(eb.eb)
# push!(metrics, [iteration_number, cost, total_pv, mean_burden, max_burden])

# Create metrics plot
metrics_plot = plot(
    metrics.iteration,
    [metrics.cost metrics.total_pv metrics.mean_burden metrics.max_burden],
    label=["Cost" "Total PV" "Mean Burden" "Max Burden"],
    xlabel="Timestep",
    ylabel="Value",
    title="System Metrics Over Time",
    linewidth=2,
    marker=:circle
)
savefig(metrics_plot, "plots_$date/metrics.png")

# Create plot for total PV generation per day
# pv_plot = plot(
#     title="Total PV Generation Per Day",
#     xlabel="Time Step",
#     ylabel="Total PV Generation (kW)",
#     legend=:outerright
# )

#for i in keys(ref_load)
# day_pv = []
#     for day in 1:days
#         # Calculate total PV for all buses at each timestep for this day, convert from p.u. to kW (baseMVA * 1000)
#           push!(day_pv,round(sum(value.(acopf_model.ext[:variables][:x_pv][5,day,:])), sigdigits=2))
#             plot!(pv_plot, 1:time_steps, day_pv, label="Day $day", linewidth=2)
#     end
# #end
# savefig(pv_plot, "plots_$date/total_pv_per_day.png")
# create individual line plots for each metric with the x axis being the iteration number

# total load plot
total_load_plot = plot(
    metrics.iteration,
    metrics.total_load,
    label="Total Load",
    xlabel="Timestep",
    ylabel="Value",
    title="Total Load Over Time",
    linewidth=2,
    marker=:circle
)
savefig(total_load_plot, "plots_$date/total_load.png")

# total pv generation
total_pv_plot = plot(
    metrics.iteration,
    metrics.total_pv,
    label="Total PV",
    xlabel="Timestep",
    ylabel="Value",
    title="Total PV Generation Over Time",
    linewidth=2,
    marker=:circle
)
savefig(total_pv_plot, "plots_$date/total_pv.png")
# Mean energy burden
mean_burden_plot = plot(
    metrics.iteration,
    metrics.mean_burden,
    label="Mean Burden",
    xlabel="Timestep",
    ylabel="Value",
    title="Mean Energy Burden Over Time",
    linewidth=2,
    marker=:circle
)
savefig(mean_burden_plot, "plots_$date/mean_burden.png")

# Max energy burden
max_burden_plot = plot(
    metrics.iteration,
    metrics.max_burden,
    label="Max Burden",
    xlabel="Timestep",
    ylabel="Value",
    title="Max Energy Burden Over Time",
    linewidth=2,
    marker=:circle
)
savefig(max_burden_plot, "plots_$date/max_burden.png")

# Plot comparison of PV and load for bus 5 on day 2
comparison_plot = plot()

# Get load data for bus 5, day 2
load_data =  [value(acopf_model.ext[:variables][:p_load][5,2,t]) for t in 1:time_steps]

# Get PV data for bus 5, day 2
pv_values = [value(acopf_model.ext[:variables][:x_pv][5,2,t]) for t in 1:time_steps]

# Plot both on same axes
plot!(comparison_plot, 1:time_steps, load_data, 
      label="Load", 
      linewidth=2, 
      marker=:circle,
      markersize=3)
plot!(comparison_plot, 1:time_steps, pv_values, 
      label="PV Generation",
      linewidth=2,
      marker=:square,
      markersize=3)

xlabel!("Time Step")
ylabel!("Power (MW)")
title!("Load vs PV Generation for Bus 5 on Day 2")
savefig(comparison_plot, "plots_$date/load_vs_pv_bus5_day2.png")
