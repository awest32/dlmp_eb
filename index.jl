"""
Compute gini coefficient of a vector x
"""
function gini(x)
    numrtr = 0
    d = length(x)
    for i =1:d
        for j =1:d
            numrtr += abs(x[i] - x[j])
        end
    end
    denmtr = (d-1)*norm(Diagonal(x)*x)^2
    return numrtr/denmtr
end
# gini calculation reference 
# https://en.wikipedia.org/wiki/Gini_coefficient
# Gini coefficient's theoretical range is from 0 (total equality) to 1 (absolute inequality).
function gini_coefficient(x)
    n = length(x)  # Number of elements in the vector
    total_sum = sum(x)  # Sum of all elements in the vector
    
    # Handle the case where total_sum is zero to avoid division by zero
    if total_sum == 0
        return 0.0  # Gini coefficient is zero if all elements are zero
    end
    
    # Double summation of absolute differences
    abs_diff_sum = 0.0
    for i in 1:n
        for j in 1:n
            abs_diff_sum += abs(x[i] - x[j])
        end
    end
    
    # Compute the Gini coefficient
    G = abs_diff_sum / (2 * n * total_sum)
    
    return G
end

"""
Computes the jain index of a vector x
If the system is fair, the Jain's fairness index will be 1. 
If the system is somewhat unfair, the Jain's fairness index will be between 1/n and 1. With k users having 1/k of the resources each, the Jain's fairness index will be 1/k.
If the system is completely unfair, the Jain's fairness index will be 1/n, where n is the number of users.
"""

function jain(x)
    numrtr = norm(x,1)^2
    denmtr = length(x)*norm(x)^2
    return numrtr/denmtr
end

