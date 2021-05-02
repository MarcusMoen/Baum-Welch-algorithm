import numpy as np


# Function for calculating the alpha variables in BW.
# Takes as its input a transition matrix A, emission matrix B, and a sequence of observations O.
# Its output value is a (T x n) matrix. Each row represent a time t and each column represent a state i.
def forward_prob(A, B, O):
    T = len(O)
    n = len(A[0])
    alpha = np.zeros((T, n)) # Makes the alpha matrix, initially consisting of just zeroes
    alpha[0] = np.full((1,n), 1/n) # Set the first row to be 1/n for each state
    for t in range(1, T): # Goes over all the rows excpet for the first
        for j in range(n): # Goes over all states
            result = 0
            for i in range(n):
                result += alpha[t-1][i] * A[i][j] * B[i][O[t]] # Does the recursive step
            alpha[t][j] = result # Update the alpha matrix for each time t and state j
    return alpha

# Is the same as forward_prob, except that it calculates the beta variables not the alpha variables.
# Takes as its input a transition matrix A, emission matrix B, and a sequence of observations O.
# Its output value is a (T x n) matrix. Each row represent a time t and each column represent a state i.
def backward_prob(A, B, O):
    T = len(O)
    n = len(A[0])
    beta = np.zeros((T, n)) # Makes the beta matrix, initially consisting of just zeroes
    beta[T-1] = np.full((1, n), 1) # Set the last row to be 1/n for each state
    for t in range(1, T): # Goes over all the rows excpet for the last
        for j in range(n): # Goes over all states
            result = 0
            for i in range(n):
                result += beta[T-t][i] * A[j][i] * B[i][O[T-t]] # Does the recursive step
            beta[T-t-1][j] = result # Update the beta matrix for each time T-t-1 and state j
    return beta


# Function for calculating the xi variables in BW.
# Takes as its input a alpha matrix alpha, beta matrix beta, transition matrix A, emission matrix B, and a sequence of observations O.
# Its output value is a (T x n x n) 3D array. Where the first dimension represent the time t, the second dimension a state i and the third dimension a state j
def xi_prob(alpha, beta, A, B, O):
    T = len(O) - 1
    n = len(A[0])
    xi = np.zeros((T, n, n)) # Makes a 3D array which represent time, state i, and state j
    for t in range(T): # Goes over all time t
        prob_obs = 0
        for k in range(n): # Goes over all states k to calculate P(O|lambda)
            prob_obs += alpha[t][k] * beta[t][k]
        for i in range(n): # Goes over all state i
            for j in range(n): # Goes over all state j
                xi[t][i][j] = (alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j]) / prob_obs# Calculate each xi_t(i,j) and stores them in the 3D array
    return xi


# Function for calculating the gamma variables in BW.
# Takes as its input a alpha matrix alpha, beta matrix beta, transition matrix A, and a sequence of observations O.
# Its output value is a (T x n) matrix. Where the first dimension represent the time t, the second dimension a state i
def gamma_prob(alpha, beta, A, O):
    T = len(O)
    n = len(A[0])
    gamma = np.zeros((T, n)) # Makes a gamma matrix
    for t in range(T): # Goes over all time t
        prob_obs = 0
        for k in range(n): # Goes over all states k to calculate P(O|lambda)
            prob_obs += alpha[t][k] * beta[t][k]
        for i in range(n):# Goes over all state i
            gamma[t][i] = (alpha[t][i] * beta[t][i]) / prob_obs# Calculate each gamma_t(i) and stores them in the gamma matrix
    return gamma


# The main part of the algorithm
# Takes as its input a first estimate transition matrix A, emission matrix B, a sequence of observations O, and number of iterations i
# Its output value is the new, estimate, of A and / or B
def baum_welch(A, B, O, i):
    n = len(A[0])
    for j in range(i):
        alpha = forward_prob(A, B, O) #Calculate a matrix of alpha probabilities
        beta = backward_prob(A, B, O) #Calculate a matrix of beta probabilities
        xi = xi_prob(alpha, beta, A, B, O) #Calculate a matrix of xi probabilities
        gamma = gamma_prob(alpha, beta, A, O) #Calculate a matrix of gamma probabilities
        for k in range(n):
            for l in range(n):
                xi_sum1 = 0
                xi_sum2 = 0
                for t in range(len(xi)):
                    xi_sum1 += xi[t][k][l] #Calculate the nominator in the estimation of the transition probabilities
                    for a in range(n):
                        xi_sum2 += xi[t][k][a] #Calculate the denominator in the estimation of the transition probabilities
                A[k][l] = xi_sum1 / xi_sum2 #Update each element of the transition matrix
                

        for k in range(n):
            for m in range(len(B[0])):
                gamma_sum1 = 0
                gamma_sum2 = 0
                for t in range(len(O)):
                    if O[t] == k:
                        gamma_sum1 += gamma[t][k] #Calculate the nominator in the estimation of the emission probabilities
                    gamma_sum2 += gamma[t][k] #Calculate the denominator in the estimation of the emission probabilities
                B[k][m] = gamma_sum1 / gamma_sum2 #Update each element of the emission matrix

    return A, B
  
  
