#!usr/bin/python

import sys
import re
import pprint
import string
import os, errno
import networkx as nx
import matplotlib.pyplot as plt

file_name = sys.argv[1];
print ("Opening File: "+file_name+"\n")
try:
	file = open(file_name,'r')
except FileNotFoundError:
	print ("ERROR: There was error in reading file ", file_name)
	sys.exit(0)

route = []
route_list = []
nameOfDataset = ""
timeForCompute = 0
weights = 200
G = nx.Graph()
pos = nx.spring_layout(G, k= 0.15, iterations=20)



for line in file.readlines():
	if "Name of Dataset" in line:
		if ":" in line:
			nameOfDataset = line.split(':')[1]
			nameOfDataset = nameOfDataset.lstrip().rstrip();
			print (nameOfDataset)
			#sys.exit(0)
	if "Time for compute" in line:
		timeForCompute = line.split(':')[1].split(' ')[1]
	if "Route" in line:
		route = list(map(int, line.split(':')[1].lstrip().rstrip().split(',')))
		route.insert(0,0)
		route.insert(len(route),0)
		print (route)
		route_list.append(route)
		route = []

# for each_tuple in route_list:
# 	temp_list = list(each_tuple)
# 	temp_list.insert(0,0)
# 	temp_list.insert(len(temp_list),0)
# 	cycle_tuple = tuple(temp_list)
# 	route_list[route_list.index(each_tuple)] = cycle_tuple

for item in route_list:
	edge_list = list(item)
	G.add_nodes_from(edge_list)
	weights = 200
	for i in range(len(edge_list)-1):
		edge_to_add_in_graph = (edge_list[i],edge_list[i+1])
		G.add_edge(*edge_to_add_in_graph, weight=weights)
print (G.nodes())
nx.draw_networkx(G,arrows=True,node_color = 'g', alpha = 0.8)
plt.axis('off')
plt.savefig(nameOfDataset+".png") # save as png
plt.show() # display
file.close()
