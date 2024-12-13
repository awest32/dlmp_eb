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
    pload_mat = CSV.File(load_file_name) |> Tables.matrix
    max_pload = 0
    pload_data = zeros(96,238)
    for col in 1:size(pload_mat,2)
        r=0
        for row in 1:3:size(pload_mat,1)
            r+=1
            pload_data[r,col] = pload_mat[row,col]
            if !isnan(pload_data[r,col])
                max_pload = max(max_pload,pload_data[r,col])
            else
                pload_data[r,col] = 0
            end
        end
    end
    # normalize the pv_gen_data to be between 0 and 1
    pload_data_normed = pload_data./max_pload
    return pload_data_normed
end


function read_pv_data(pv_file_name, ghi_file_name)
    ghi_data = DataFrame(CSV.File(ghi_file_name))
    max_ghi = maximum(ghi_data[:, 1])
    ghi_data_normed = ghi_data[:, 1] ./ max_ghi
    ghi_data_normed_new = ghi_data_normed[1:3:Int(length(ghi_data_normed)/5),1]

    pv_data = DataFrame(CSV.File(pv_file_name))
    max_pv = maximum(pv_data[:, 1])
    pv_data_normed = pv_data[:, 1] ./ max_pv
    pv_data_normed_new = pv_data_normed[1:3:Int(length(pv_data_normed)/5),1]
    return pv_data_normed, ghi_data_normed
end
current_dir = @__DIR__
pload_data_normed = read_load_data(current_dir*"/Pload_p.csv")
pv_data_normed, ghi_data_normed = read_pv_data(current_dir*"/pv_gen.csv", current_dir*"/ghi.csv")