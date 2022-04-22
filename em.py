import random
import math
import numpy as np
import matplotlib.pyplot as plt

def em_single(training_data, pi_1, pi_2, p, q, r):
    '''
    :pi_1,pi_2: 抽到硬币A、B的概率p(z)
    :p,q,r: 硬币A\B\C正面的概率p(y|z)
    :training_data:训练数据010101
    :return:
    '''
    n = len(training_data)
    pi_1_new = 0.0
    pi_2_new = 0.0
    p_new = 0.0
    q_new = 0.0
    r_new = 0.0
    za_list = []     
    zb_list = []     
    zc_list = []    

    pi_3 = 1 -pi_1 -pi_2
    print(pi_1,pi_2,p,q,r)

    for i, v in enumerate(training_data):
        za = pi_1 * p**v * (1-p)**(1-v) 
        zb = pi_2 * q**v * (1-q)**(1-v) 
        zc = pi_3 * r**v * (1-r)**(1-v)

        za_list.append( za/(za+zb+zc) )
        zb_list.append( zb/(za+zb+zc) )
        zc_list.append( zc/(za+zb+zc) )

    sum_L = 0
    for i, v in enumerate(training_data):
        media_1 = za_list[i] * math.log( pi_1* (p**v * (1-p)**(1-v)) / za_list[i], math.e)
        media_2 = zb_list[i]* math.log( pi_2* (q**v * (1-q)**(1-v)) / zb_list[i], math.e)
        media_3 = zc_list[i] * math.log( pi_3* (r**v * (1-r)**(1-v)) / zc_list[i], math.e)
        sum_L += media_1 + media_2 + media_3
    print(sum_L)

    pi_1_new = sum(za_list)/n 
    pi_2_new = sum(zb_list)/n 

    print(pi_1_new,pi_2_new)

    sum1 = 0 ; sum2 = 0 ; sum3 = 0
    for i, v in enumerate(training_data):
        sum1 += za_list[i]*v
        sum2 += zb_list[i]*v
        sum3 += zc_list[i]*v
    p_new = sum1 / sum(za_list)   
    q_new = sum2 / sum(zb_list)   
    r_new = sum3 / sum(zc_list)
    print(p_new,q_new,r_new)

    return [pi_1_new, pi_2_new, p_new, q_new, r_new, sum_L]

    
def traindata_generate(pi_1, pi_2, p, q, r, n):
    training_data = []
    for i in range(n):
        r1 = random.random()
        r2 = random.random()
        if r1 <= pi_1:
            if r2 <= p:
                training_data.append(1)
            else:
                training_data.append(0)
        if (r1 > pi_1) and (r1 <= pi_1 + pi_2):
            if r2 <= q:
                training_data.append(1)
            else:
                training_data.append(0)
        if r1 > pi_1 + pi_2:
            if r2 <= r:
                        training_data.append(1)
            else:
                        training_data.append(0)
    return training_data


def em(training_data, pi_1, pi_2, p, q, r, tol = 1e-6, iterations=5):
    likelihood = []
    parameters = [[pi_1,pi_2,p,q,r]]
    for i in range(iterations):
        [pi_1_new, pi_2_new, p_new, q_new, r_new, sum_L] = em_single(training_data, pi_1, pi_2, p, q, r)
        pi_1 = pi_1_new
        pi_2 = pi_2_new
        p = p_new
        q = q_new
        r = r_new
        parameters.append([pi_1,pi_2,p,q,r])
        likelihood.append(sum_L)
    print([pi_1,pi_2,p,q,r], iterations)
    return ([pi_1,pi_2,p,q,r], iterations, likelihood, parameters)

    
pi_1_true = 0.2
pi_2_true = 0.3
p_true = 0.2
q_true = 0.5
r_true = 0.8
n_true = 200
training_data = traindata_generate(pi_1_true, pi_2_true, p_true, q_true, r_true, n_true)
_, iterations, likelihood, parameters = em(training_data, pi_1=0.23, pi_2=0.32, p=0.18, q=0.46, r=0.78, tol = 1e-6, iterations=5)
plt.figure(1)
plt.title('Maximum Likelihood Estimation (Objective function)')
plt.plot(np.arange(iterations), likelihood)
plt.xlabel("iterations") 
plt.ylabel("maximum likelihood estimation value")   
plt.savefig('./pic/likelihood.png')
plt.show()
pi_1_list = []
pi_2_list = []
p_list = []
q_list = []
r_list = []
for parameter in parameters:
    pi_1_list.append(parameter[0])
    pi_2_list.append(parameter[1])
    p_list.append(parameter[2])
    q_list.append(parameter[3])
    r_list.append(parameter[4])
plt.figure(2)
plt.title('S')
plt.plot(np.arange(iterations+1), pi_1_list, c='gray', label='S1')
plt.plot(np.arange(iterations+1), pi_2_list, c='red', label='S2')
plt.legend(['S1', 'S2'])
plt.xlabel("iters") 
plt.ylabel("value")   
plt.axhline(y=pi_1_true, ls=':', c='gray') 
plt.axhline(y=pi_2_true, ls=':', c='red')
plt.savefig('./pic/parameters_pi.png')
plt.show()
plt.figure(3)
plt.plot(np.arange(iterations+1), p_list, c='gray', label='p')
plt.plot(np.arange(iterations+1), q_list, "r-", label='q')
plt.plot(np.arange(iterations+1), r_list, "b-", label='r')
plt.legend(['p', 'q', 'r'])
plt.title('pqr')
plt.xlabel("iters") 
plt.ylabel("value")   
plt.axhline(y=p_true, ls=':', c='gray') 
plt.axhline(y=q_true, ls=':', c='red')
plt.axhline(y=r_true, ls=':', c='blue')
plt.savefig('./pic/parameters_pqr.png')
plt.show()