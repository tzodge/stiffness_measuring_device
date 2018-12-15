
'''
Use the following commands in a terminal if it shows error
"gurobi.sh not found"  when you run gurobi.sh
>> cd /opt/gurobi752/linux64


#>> export GUROBI_HOME="/opt/gurobi752/linux64"
#>> export PATH="${PATH}:${GUROBI_HOME}/bin"
#>> export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"

then 
>> gurobi.sh 
sould open a gurobi shell

you will have to operate in the same terminal to run a
gurobi program

>> cd  /home/biorobotics/cpp_test/newLibTest 

'''

from gurobipy import *
import numpy as np 


#http://www.gurobi.com/documentation/8.0/refman/py_model_addconstrs.html i !=j , i!=-j+1 

Ns = 10
a = [-9,-15,-5]
tstar = np.reshape(a,(3,1))
M =  np.random.rand(3,Ns)
S = M - tstar

print S.shape
print M.shape

# Model
m = Model("GurobiTest1")
# n1= 1*np.array((tstar.shape[0]))
n1 = 3
n2 = 10
# n2= Ns*np.array((tstar.shape[1])) 
print n1
print n2
print ""
# The objective is to minimize the costs

alpha = m.addVars(n1,n2)
phi =  m.addVars(n1,1)

print alpha
print "............."

t = m.addVars(3,1 , lb=-GRB.INFINITY, ub = GRB.INFINITY)
print t
print "............."

# phi = m.addVars((1*np.array((tstar.shape[0])), Ns*np.array((tstar.shape[1]))) )

# print alpha
# x = m.addVars(10,3)
# print x
# # print x
# # Using looping constructs, the preceding statement would be:
# #
# # m.setObjective(sum(buy[f]*cost[f] for f in foods), GRB.MINIMIZE)

# # Nutrition constraints


# k= 0
# for i in range (0,10):
# 	for j in range (0,3):

# 		m.addConstr(x[i,j] <= 10 )
# 		print k 


for i  in range (0,3):	
	for j in range (0,10):
		m.addConstr( alpha[i,j] - (S[i,j] + t[i,0] - M[i,j]) >=  0	)
		m.addConstr( alpha[i,j] + (S[i,j] + t[i,0] - M[i,j]) <=  0	)
		
		print S[i,j]
		m.addConstr(phi[i,0] >= 0)
		m.addConstr(alpha[i,j] >= 0)
		m.addConstr(phi[i,0] >= alpha[i,0]+alpha[i,1]+alpha[i,2])
		# m.addConstr(np.linalg.norm(alpha) <= 2)
		# ObjFunc = ObjFunc + phi[i,j]

		print i,j
# ObjFunc = ObjFunc/Ns
# print phi.sum()
m.setObjective((phi.sum()/Ns) , GRB.MINIMIZE)

# m.addConstr(phi[i,j] >= 0 for i  in range (0,Ns))
# m.addConstr(alpha[i] >= 0 for i  in range (0,Ns))


# Using looping constructs, the preceding statement would be:
#
# for c in categories:
#  m.addRange(
#     sum(nutritionValues[f,c] * buy[f] for f in foods), minNutrition[c], maxNutrition[c], c)

# Solve


m.optimize()

if m.Status == GRB.OPTIMAL:
   for i in range (0,3):
   		print t[i,0]

# printSolution()

# print phi
# print alpha