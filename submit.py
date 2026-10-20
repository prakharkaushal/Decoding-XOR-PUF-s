import numpy as np
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel

# You are allowed to import any submodules of sklearn e.g. metrics.pairwise to construct kernel Gram matrices
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_kernel, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here

################################
# Non Editable Region Starting #
################################
def my_kernel( X1, Z1, X2, Z2 ):
################################
#  Non Editable Region Ending  #
################################

    # Use this method to compute Gram matrices for your proposed kernel
    # Your kernel matrix will be used to train a kernel ridge regressor
    # best_d=2,best_c=0.1
    X1 = np.asarray(X1).reshape(-1, 1)   
    X2 = np.asarray(X2).reshape(-1, 1)   
    Z1 = np.asarray(Z1)                  
    Z2 = np.asarray(Z2)                  
    dot_z = Z1 @ Z2.T                    
    K_z = (dot_z + 0.1) ** 2
    
    K_x = X1 @ X2.T              
    G = K_x * K_z + 1.0

    return G


################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################

    # Use this method to invert a PUF linear model to get back delays
    # w is a single 1089-dim vector (last dimension being the bias term)
    # The output should be eight 32-dimensional vectors
    
    W = w.reshape(33, 33)
    
    U, S, Vt = np.linalg.svd(W)
    
    sigma = S[0]
    
    u_weights = U[:, 0] * np.sqrt(sigma)
    v_weights = Vt[0, :] * np.sqrt(sigma)
    
    def solve_single_apuf(weights):
        # w_0 = alpha_0, w_i = alpha_i + beta_{i-1}, w_32 = beta_31 is underdetermined;
        # fixing beta_0..beta_30 = 0 gives alpha_i = w_i and beta_31 = w_32
        alpha = weights[:32].copy()
        beta = np.zeros(32)
        beta[31] = weights[32]

        diff_pq = alpha + beta
        diff_rs = alpha - beta

        p = np.maximum(diff_pq, 0)
        q = np.maximum(-diff_pq, 0)
        r = np.maximum(diff_rs, 0)
        s = np.maximum(-diff_rs, 0)
        
        return p, q, r, s

    p1, q1, r1, s1 = solve_single_apuf(u_weights)
    p2, q2, r2, s2 = solve_single_apuf(v_weights)
    
    return p1, q1, r1, s1, p2, q2, r2, s2

