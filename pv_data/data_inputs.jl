#test the read in of the pv data 
using CSV, Plots
import GR
using Dates 
using DataFrames
using LinearAlgebra
using Printf
using JLD2
using Statistics

function read_load_data(load_file_name)
    # Read in the pv generation data for
    # The columns are the days (238)
    # The rows are the 5 minute intervals (288)
    pload_mat = DataFrame(CSV.File(load_file_name))# |> Tables.matrix
    max_pload = 1
    r,c = size(pload_mat)
    pload_data = zeros(r,c)
    for col in 1:c
        for row in 1:r
            pload_data[row,col] = pload_mat[row,col]
            if !isnan(pload_data[row,col])
                max_pload = max(max_pload,pload_data[r,col])
            # else
            #     pload_data[r,col] = 0
            end
        end
    end
    # normalize the pv_gen_data to be between 0 and 1
    pload_data_normed = pload_data./max_pload
    return pload_data_normed, pload_data
end


function read_pv_data(pv_file_name, ghi_file_name)
    ghi_data = DataFrame(CSV.File(ghi_file_name))
    max_ghi = maximum(ghi_data[:, 1])
    ghi_data_normed = ghi_data[:, 1] ./ max_ghi

    pv_data = DataFrame(CSV.File(pv_file_name))
    max_pv = maximum(pv_data[:, 1])
    pv_data_normed = pv_data[:, 1] ./ max_pv
    return pv_data_normed, ghi_data_normed
end
# current_dir = @__DIR__
# pload_data_normed,pload_data = read_load_data(current_dir*"/Pload_p.csv")
# pv_data_normed, ghi_data_normed = read_pv_data(current_dir*"/pv_gen.csv", current_dir*"/ghi.csv")

function extract_small_amount_of_time(time_frame,days,load_data, pv_data,data_resolution)
    #extract the small time frame from the data
    seasonal_time_daily = Int(floor((238)/3.5))
    seasonal_time_intervals = seasonal_time_daily*24*(60/data_resolution)
    seasonal_time_intervals = Int(seasonal_time_intervals)
    #load_days = size(load_data)[2]
    #time_frame_data_daily = LinRange(1, load_days, seasonal_time_daily)
    #time_frame_data_intervals = LinRange(1, load_days, seasonal_time_intervals)
    test_load = DataFrame(day = Int[],load = Float64[])
    test_pv = DataFrame(day = Int[],pv = Float64[])
    day = 0
    for i in 1:time_frame#seasons
        println("season number: ", i)
        for j in 1:days
            day += 1
            println("day: ", j)
            println("seasonal_time_daily: ", (seasonal_time_daily*(i-1) +1*j))
            test_days = load_data[:, (1*j)+(seasonal_time_daily*(i-1))]
            println("seasonal_time_intervals: ", day*seasonal_time_intervals+1)
            test_ints = pv_data[(1*j)+(seasonal_time_intervals*(i-1)):(1*j) + (seasonal_time_intervals*(i-1)) + Int(24*(60/data_resolution))]
            for z in 1:Int(24*(60/data_resolution))
                #rint("Interval: ", z)
                push!(test_load, [day, test_days[z]])
                push!(test_pv, [day, test_ints[z]])
                #seasonal_time_daily = seasonal_time_daily * time_frame
                #seasonal_time_intervals = seasonal_time_intervals * time_frame
            end
        end
    end
    return test_load, test_pv
end
#extr_load, extr_pv = extract_small_amount_of_time(2,2,pload_data_normed, pv_data_normed, 5)


