#test the read in of the pv data 
using CSV, Plots
import GR
using Dates 
using DataFrames
using LinearAlgebra
using Printf
using JLD2
using Statistics

function read_load_data()
    current_dir = @__DIR__
    # Read in the pv generation data for
    # The columns are the days (238)
    # The rows are the 5 minute intervals (288)
    pload_mat = CSV.File(current_dir*"/Pload_p.csv") |> Tables.matrix
    max_pv = 0
    pv_gen_data = zeros(96,238)
    for col in 1:size(pload_mat,2)
        r=0
        for row in 1:3:size(pload_mat,1)
            r+=1
            pv_gen_data[r,col] = pload_mat[row,col]
            if !isnan(pv_gen_data[r,col])
                max_pv = max(max_pv,pv_gen_data[r,col])
            end
        end
    end
    # normalize the pv_gen_data to be between 0 and 1
    pv_gen_data_normed = pv_gen_data./max_pv
    return pv_gen_data,max_pv, pv_gen_data_normed
end

pv_gen_data,max_pv,pv_gen_data_normed = read_pv_data()