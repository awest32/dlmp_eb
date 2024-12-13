#test the read in of the pv data 
using CSV, Plots
import GR
using Dates 
using DataFrames
using LinearAlgebra
using Printf
using JLD2
using Statistics

current_dir = @__DIR__
ghi_file_name = current_dir*"/ghi.csv"
ghi_data = DataFrame(CSV.File(ghi_file_name))
max_ghi = maximum(ghi_data[:, 1])
ghi_data_normed = ghi_data[:, 1] ./ max_ghi
ghi_data_normed_new = ghi_data_normed[1:3:length(ghi_data_normed),1]

pv_file_name = current_dir*"/pv_gen.csv"
pv_data = DataFrame(CSV.File(pv_file_name))
max_pv = maximum(pv_data[:, 1])
pv_data_normed = pv_data[:, 1] ./ max_pv