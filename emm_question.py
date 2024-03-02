import numpy as np
from functools import lru_cache
from scipy.integrate import quad

################################################################
##### Helper functions (DO NOT CHANGE)
################################################################

@lru_cache  # Makes things go fast
def normalise_expontial_family(sufstat, eta):
    unnorm_prob = lambda z: np.exp(sufstat(z) @ np.array(eta))
    Z, err = quad(unnorm_prob, -np.inf, np.inf)
    return float(Z)

def exponential_family_pdf(x, sufstat, eta):
    # Input Shapes: (1,), None, (M,)
    # sufstat designates the sufficient statistic map for the exponential
    # family, taking values in (1,) to (M,).
    unnorm_prob = lambda z: np.exp(sufstat(z) @ eta)
    eta = eta.squeeze()
    Z = normalise_expontial_family(sufstat, tuple(eta))
    prob = unnorm_prob(x) / Z # Here Z = exp(-psi(eta))
    
    return prob

################################################################
##### EMM Question Code
################################################################

def weighted_probs(data, pi, eta, sufstat, N, K):
    # Input Shapes: (N,), (K,1), (K,m), None, None
    # Should implement pi_k * q(x_n|eta_k) for each n, k, and thus return shape
    # should be (N,K). You should use exponential_family_pdf as defined above.
    # Note: sufstat(x) = u(x).
    # Works for scalars ((1,) -> (2,)); and 1D arrays ((N,) -> (N, 2)).
    ### CODE HERE ###
    weight_matrix = np.ones(shape = [N, K])
    for i in range(N):
        for a in range(K):
            implement_number = pi[a,0] * exponential_family_pdf(data[i], sufstat, eta[a, :])
            weight_matrix[i,a] = implement_number
    return weight_matrix

def e_step_EMM(data, pi, eta, sufstat, N, K):
    # Input Shapes: (N,), (K,1), (K,m), None, None
    # Should implement gamma_nk for each n, k; and thus return shape should be (N,K).
    # Note: sufstat(x) = u(x).
    # This works for scalars ((1,) -> (2,)); and 1D arrays ((N,) -> (N, 2)).
    # It should use weighted_probs.
    ### CODE HERE ###
    gamma_matrix = np.ones(shape = [N, K])
    matix_of_weight_probs = weighted_probs(data, pi, eta, sufstat, N, K)
    sum_of_each_row = np.sum(matix_of_weight_probs, axis=1)
    for i in range(N):
        for a in range(K):
            numerator = matix_of_weight_probs[i,a]
            denominator = sum_of_each_row[i]
            elements = numerator / denominator
            gamma_matrix[i,a] = elements
    return gamma_matrix

def m_step_EMM(data, gamma, sufstat, exp_to_nat, N, K):
    # Input Shapes: (N,D), (N,K), None, None, None
    # Should implement updates for pi, Eta, and return them in that order.
    # exp_to_nat is a function which converts the expectation parameter to
    # natural parameter. This only works dimensions (2,) -> (2,).
    # Note: sufstat(x) = u(x).
    # This works for scalars (1,) -> (2,); and 1D arrays (N,) -> (N, 2).
    # Return shapes should be (K,1), (K,m).
    ### CODE HERE ###
    # obtain sufficient stat
    xn = sufstat(data)
    m = xn.shape[1]
    
    new_pi = np.ones(shape = [K, 1])
    new_eta = np.zeros(shape = [K, m])
    
    # obtain Nk
    nk = np.zeros(shape = [K, 1])
    for i in range(K):
        for j in range(N):
            nk[i] += gamma[j,i]
    # maximise new_pi
    for i in range(K):
        new_pi[i] = nk[i]/N
        
    # maximise new_eta
    for i in range(K):
        one_lmbd = 0
        for j in range(N):
            one_lmbd += xn[j]*gamma[j,i]
            
        one_lmbd /= nk[i]
        one_eta = exp_to_nat(one_lmbd)
        for l in range(m):
            new_eta[i][l] = one_eta[l] 
           
    return new_pi, new_eta # (K,1), (K,m)
