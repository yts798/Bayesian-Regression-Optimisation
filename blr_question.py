import numpy as np

################################################################
##### BLR Question Code
################################################################

def single_EM_iter_blr(features, targets, alpha_i, beta_i):
    # Given the old alpha_i and beta_i, computes expectation of latent variable w: M_n and S_n,
    # and using that computes the new alpha and beta values.
    # Should return M_n, S_n, new_alpha, new_beta in that order, with return shapes (M,1), (M,M), None, None
    ### CODE HERE ###
    # extract dimension
    (N, M) = features.shape
    
    # Expectation
    # solve for sn
    inv_sn = alpha_i * np.eye(M) + beta_i * np.dot(np.transpose(features), features)
    sn = np.linalg.inv(inv_sn)
    
    # solve for mn
    phi_t = np.dot(np.transpose(features), targets)
    mn = beta_i * np.dot(sn, phi_t) 
    
    # Update E1 and E2 
    E1 = np.dot(np.transpose(mn),mn)[0][0] + np.trace(sn)
    phi_sn = np.dot(features, sn) 
    norm_2 = np.linalg.norm((targets - np.dot(features, mn)), 2)
    E2 = np.power(norm_2, 2) + np.trace(np.dot(np.transpose(features), phi_sn))
    
    # Maximisation
    # maximise new_alpha
    new_alpha = M/E1
    
    # maximise new_alpha
    new_beta = N/E2
    
    return mn, sn, new_alpha, new_beta # (M,1), (M,M), None, None