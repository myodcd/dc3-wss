import os
import pickle
import sys


#sys.path.append('"C:\\Users\\mtcd\\Documents\\Codes\\dc3-wss\\dc3\\datasets\\dc-wss\\data_system.py"')
from data_system import data_system




filepath = os.path.join('datasets', 'dc-wss', 'dc_wss_dataset_dc_6')

with open(filepath, 'rb') as f:
    problem = pickle.load(f)
    

x = [3,4,8,9,13,1,1,3,1,0.5]

d = problem.d
pumps = problem.pumps
timeInc = problem.timeInc
pipes = problem.pipes
tanks = problem.tanks
valves = problem.valves



obj_fn = problem.Cost(x, d, pumps, tanks, pipes, valves, timeInc)



g_TempLog = problem.g_TempLog(x, d)

gT = problem.g_TempLog(x, d)

g_TempLof_dist = problem.g_TempLog_dist(x, d)

jac_TempLog = problem.jac_TempLog(x, d)

print(x)
print(obj_fn)
print('gt ' , gT)
print('g_TempLog',  g_TempLog)

print(g_TempLof_dist)

print('jac_TempLog', jac_TempLog)