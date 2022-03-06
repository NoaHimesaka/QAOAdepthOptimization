"""
The PG model
First 100 iterations are calculated using the regularized model, and after that the regularization term has been removed
"""
import pyqpanda as q
import numpy as np
from pyqpanda import *
import pandas as pd
from matplotlib import  pyplot as plt
import time
import os


"""
Definition of the instance
"""
#probelm = {'Z0 Z5':0.64,'Z0 Z6':0.37,'Z1 Z6':0.86,'Z1 Z7':0.74,'Z2 Z3':0.13,'Z2 Z5':0.59,'Z2 Z6':0.17,'Z2 Z7':0.63,'Z3 Z6':0.84,'Z3 Z7':0.44,'Z4 Z5':0.45,'Z4 Z7':0.78,'Z5 Z6':0.21}
probelm = {'Z0 Z4':0.73, 'Z0 Z5':0.33, 'Z0 Z6':0.5,'Z1 Z4':0.69,'Z1 Z5':0.36,'Z2 Z5':0.88,'Z2 Z6':0.58,'Z3 Z5':0.67,'Z3 Z6':0.43}
#probelm = {'Z0 Z1':0.21,'Z0 Z2':0.44,'Z0 Z3':0.64,'Z0 Z4':0.33,'Z1 Z2':0.78,'Z1 Z3':0.37,'Z1 Z4':0.55,'Z3 Z4':0.02}

"""
Parameter setting
"""

step = 7 # the depth of quantum circuit
iter = 300 # iterations
lr = 0.0065 # learning rate
disturb = 0.000001 # Parameter in the finite difference method
st = 0.432 # lambda



time_now = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
path = "./result"+'/'+time_now
print("path = ",path)
if not os.path.exists(path):
    os.makedirs(path)

"""
The definition of the quantum circuit
"""
def oneCircuit(qlist, Hamiltonian, beta, gamma):
    vqc = q.VariationalQuantumCircuit()
    for i in range(len(Hamiltonian)):
        tmp_vec = []
        item = Hamiltonian[i]
        dict_p = item[0]
        for iter in dict_p:
            if 'Z' != dict_p[iter]:
                pass
            tmp_vec.append(qlist[iter])
        coef = item[1]

        if 2 != len(tmp_vec):
            pass
        vqc.insert(q.VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))
        vqc.insert(q.VariationalQuantumGate_RZ(tmp_vec[1], 2 * gamma * coef))
        vqc.insert(q.VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))

    for j in qlist:
        vqc.insert(q.VariationalQuantumGate_RX(j, 2 * beta))

    return vqc

Hp = q.PauliOperator(probelm)
'''
{Z0 Z4 : 0.730000
Z0 Z5 : 0.330000
Z0 Z6 : 0.500000
Z1 Z4 : 0.690000
Z1 Z5 : 0.360000
Z2 Z5 : 0.880000
Z2 Z6 : 0.580000
Z3 Z5 : 0.670000
Z3 Z6 : 0.430000
}
'''
qubit_num = Hp.getMaxIndex()
machine = q.init_quantum_machine(QMachineType.CPU_SINGLE_THREAD)
qlist = machine.qAlloc_many(qubit_num)

"""
Initial value of beta and gamma
"""
beta = q.var(0.3*np.ones((step, 1), dtype='float64'), True)
gamma = q.var(0.3*np.ones((step, 1), dtype='float64'), True)

vqc = q.VariationalQuantumCircuit()
for i in qlist:
    vqc.insert(q.VariationalQuantumGate_H(i))

for i in range(step):
    vqc.insert(oneCircuit(qlist, Hp.toHamiltonian(1), beta[i], gamma[i]))

def norm(args):
    sum = 0
    sum = var(sum)
    for i in range(len(eval(args))):
        sum += var(abs(eval(args[i])))
    return sum

def indeff(beta,gamma,lamda):
    sum = lamda*(norm(beta)+norm(gamma))
    return sum

# positive is used to return a F(theta+phi)
def positive(beta,gamma,num,qlist,Hp,machine,disturb):
    c=np.vstack((eval(beta),eval(gamma)))
    c[num][0] += disturb
    beta, gamma = np.split(c,2)
    beta = var(beta)
    gamma = var(gamma)

    vqc = q.VariationalQuantumCircuit()
    for i in qlist:
        vqc.insert(q.VariationalQuantumGate_H(i))
    for i in range(step):
        vqc.insert(oneCircuit(qlist, Hp.toHamiltonian(1), beta[i], gamma[i]))
    loss = eval(q.qop(vqc, Hp, machine, qlist))
    return loss

def negative(beta , gamma, num, qlist, Hp, machine,disturb):
    c=np.vstack((eval(beta),eval(gamma)))
    c[num][0] -=disturb
    beta,gamma = np.split(c,2)
    beta = var(beta)
    gamma = var(gamma)
    vqc = q.VariationalQuantumCircuit()
    for i in qlist:
        vqc.insert(q.VariationalQuantumGate_H(i))
    for i in range(step):
        vqc.insert(oneCircuit(qlist, Hp.toHamiltonian(1), beta[i], gamma[i]))
    loss = eval(q.qop(vqc, Hp, machine, qlist))
    return loss

def cal_grad(step,beta,gamma,qlist,machine,disturb):
    gradient = np.zeros((2*step,1))
    for i in range(2*step):
        c=np.vstack((eval(beta),eval(gamma)))
        if c[i][0]==0:
            gradient[i]=0
        else:
            gradient[i] = (positive(beta,gamma,i,qlist,Hp,machine,disturb)-negative(beta,gamma,i,qlist,Hp,machine,disturb))/(2*disturb)
    return gradient

def input_prox(beta,gamma,lr,gradient):
    c = np.vstack((eval(beta),eval(gamma)))
    c = c - lr*gradient
    return c


def soft_thresholding(input,landa,lr):
    if input>landa*lr:
        result = input - landa*lr
    elif input <-landa*lr:
        result = input + landa*lr
    else:
        result = 0
    return result

def hard_thresholding(input,landa):
    if input>landa:
        result = input
    elif input <-landa:
        result = input
    else:
        result = 0
    return result

def l2normsquare(input,step):
    sum=0
    for i in range(2*step):
        sum=sum+input[i]*input[i]
    return sum

calc_loss = np.zeros((iter,1))
beacon = 0
row = np.zeros((iter,2*step))
k_cal = []
loss_cal = np.zeros(iter)
loss_list=np.zeros(iter)
"""
Iteration process
"""
for i in range(iter):
    gradient = cal_grad(step, beta, gamma, qlist, machine, disturb)# calculate the gradient
    input = input_prox(beta, gamma, lr, gradient)# x=x-lr*gradient
    c = np.zeros((2 * step, 1))
    for j in range(2 * step):
        c[j] = soft_thresholding(input[j][0], st,lr)

    # 100 is the iterations that apply regularization term
    if (i<100):
        beta, gamma = np.split(c, 2)  # beta, gamma after shrinkage
    elif(i>=100):
        beta, gamma = np.split(input, 2)
    beta_list = beta.tolist()
    gamma_list = gamma.tolist()

    """
    Record of result 
    """
    for iii in range(step):
        row[i][iii] = beta[iii]
        row[i][iii + step] = gamma[iii]
    data_df = pd.DataFrame(row)
    writer1 = pd.ExcelWriter(path+'/prox.xlsx')
    data_df.to_excel(writer1, 'page_1', float_format='%.5f', header=False, index=False)
    writer1.save()
    zero = []
    for k in range(step):
        if beta_list[k][0] == 0:
            zero.append(k)
            k_cal.append(k)

        if gamma_list[k][0] == 0:
            zero.append(k)
            k_cal.append(k)
    beta = var(beta)
    gamma = var(gamma)
    for j in qlist:
        vqc.insert(q.VariationalQuantumGate_H(j))
    for k in range(step):
        vqc.insert(oneCircuit(qlist, Hp.toHamiltonian(1), beta[k], gamma[k]))

    loss = eval(q.qop(vqc, Hp, machine, qlist))
    loss_cal[i]=loss
    loss_df = pd.DataFrame(loss_cal)
    writer3 = pd.ExcelWriter(path+'/loss.xlsx')
    loss_df.to_excel(writer3, 'page_1', float_format='%.5f', header=False, index=False)
    writer3.save()
    calc_loss[i][0] = loss
    print(f"iter :{i}", "loss :", "\n", loss, "\n", "beta :", "\n", eval(beta), "\n", "gamma ", "\n", eval(gamma))
    print(f"lr = {lr}")
print("path = ", path)
#print("k = ",k_cal)
k_cal = np.array(k_cal)
k_df = pd.DataFrame(k_cal)
writer4 = pd.ExcelWriter(path+'/k.xlsx')
k_df.to_excel(writer4, 'page_1', float_format='%.5f', header=False, index=False)
writer4.save()

print(f"lr = {lr}; soft-thresholding = {st}")

plt.xlabel("iter")
plt.ylabel("loss")
plt.title(" proximal gradient descent")
plt.plot(calc_loss)
plt.show()



prog = q.QProg()
circuit = vqc.feed()
prog.insert(circuit)
q.directly_run(prog)

result = q.quick_measure(qlist, 10000)
print(result)