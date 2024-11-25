#test the read in of the pv data 
using CSV, Plots
import GR
using Dates 
using DataFrames
using LinearAlgebra
using Printf
using JLD2
using Statistics

function read_pv_data()
    current_dir = @__DIR__
    # Read in the pv generation data for
    # The columns are the days (238)
    # The rows are the 5 minute intervals (288)
    pload_mat = CSV.File(current_dir*"/Pload_p.csv") |> Tables.matrix


    pv_gen_data = zeros(96,238)
    for col in 1:size(pload_mat,2)
        r=0
        for row in 1:3:size(pload_mat,1)
            r+=1
            pv_gen_data[r,col] = pload_mat[row,col]
        end
    end
    return pv_gen_data
end
