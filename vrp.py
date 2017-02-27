# @Author: Prasad Pandit and Radhika Mandlekar
# @Date:   26-Feb-2017
# @Email:  prasad@pdx.edu & radhika@pdx.edu
# @Last modified by:   Prasad Pandit
# @Last modified time: 27-Feb-2017

import os
import csv
import sys
import networkx as nx
import matplotlib.pyplot as plt
cost_matrix = [
		[0,12,11,7,10,10,9,8,6,12],
		[12,0,8,5,9,12,14,16,17,22],
		[11,8,0,9,15,17,8,18,14,22],
		[7,5,9,0,7,9,11,12,12,17],
		[10,9,15,7,0,3,17,7,15,18],
		[10,12,17,9,3,0,18,6,15,15],
		[9,14,8,11,17,18,0,16,8,16],
		[8,16,18,12,7,6,16,0,11,11],
		[6,17,14,12,15,15,8,11,0,10],
		[12,22,22,17,18,15,16,11,10,0]
		]

G = nx.Graph()

row_elements = 10
#for row in range(row_elements):
#	for column in range(row_elements):
#		print cost_matrix[row][column]
#	print "End of Row "+str(row)
#sys.exit()

demand_dict = {1:10,2:15,3:18,4:17,5:3,6:5,7:9,8:4,9:6}
#print demand_dict[3]
#print demand_dict

vehicle_capacity = 40

#Step1: Savings Calculation Sij = C0i +C0j - Cij
#i and j are indices for cities
#C represent value corresponding to cost matrix
#savings_matrix = [[0 for x in range(row_elements)] for y in range(row_elements)]
#savinigs_matrix[row_elements][row_elements]
#all_savings = []

savings_list = []
temp_list = []
for i in range(row_elements):
	for j in range(row_elements):
		if i != j and i !=0 and j != 0  and j>i :
			temp_list.append((i))
			temp_list.append((j))
			value = cost_matrix[0][i] + cost_matrix[0][j] - cost_matrix[i][j]
			temp_list.append(value)
			#savings_matrix[i][j] = value
			#all_savings.append(value)
			tuple_of_i_j_saving = tuple(temp_list)
			savings_list.append(tuple_of_i_j_saving)
			temp_list = []
		#else :
		#	savings_matrix[i][j] = 0
			#print '---------' + str(i) + ' j : ' + str(j)
			#print str(cost_matrix[0][i]) + '+ ' + str(cost_matrix [0][j]) + ' + ' + str(cost_matrix[i][j]) + ' = ' +str( value )
	#print '*******************************************'
#for e in savings_list :
#	print e

# Step 2: Savings List in Decreasing Order
savings_list  = sorted(savings_list, key=lambda element: element[2], reverse = True)

# Step 3	: Join Cycles
# Description	: Take edge[i,j] from top of savings list, join two seperate cycles
# 		  with edge[i,j] if
# 		  			1. Nodes belong to seperate cycles
# 					2. Max capacity of vehicle is not exceeded
#					3. i & j are first or last customer on the cycle

result_list = []
result_dict = {}

for i in range(row_elements-1):
	result_dict[i+1] = 0

#print (result_dict)

i = 0
exist_i = 0
exist_j = 0
index_of_tuple_for_i = 0
index_of_tuple_for_j = 0


for saving in savings_list:
	edge_i = 	 saving[0]
	edge_j = 	 saving[1]
	i += 1
	#check 1 : nodes belong to separate cycle
	print ("\nEdges : " + str(edge_i) + " , " + str(edge_j))
	#check 1 : Capacity Constraint
	#check 2 :  nodes belong to separate cycle

	if result_list:
		if demand_dict[edge_i] + demand_dict[edge_j] <= vehicle_capacity :
			print ("Iteration No.: " + str(i) + " for " + str(edge_i) + "," + str(edge_j))
			print ("Capacity Constraints Not Violated in if")
			if result_dict[edge_i]==1 and result_dict[edge_j]==0:
				for each_route in result_list:
					if edge_i in each_route:
						length_of_list = len(each_route)
						total_demand = 0
						total_demand += demand_dict[edge_j]
						for each_cap in each_route:
							total_demand += demand_dict[each_cap]
						if total_demand <= vehicle_capacity:
								index_of_tuple_for_i = each_route.index(edge_i)
								if index_of_tuple_for_i == 0 or index_of_tuple_for_i == length_of_list-1:
									route = list(each_route)
									route.insert(length_of_list, edge_j)
									cycle_tuple  = tuple(route)
									result_dict[edge_j] = 1
									index_of_result_tuple = result_list.index(each_route)
									result_list[index_of_result_tuple] = cycle_tuple
								else:
									print ("Can't add the "+ str(edge_j) +" node as " + str(edge_i) +" is intermediate node")
						else:
							print ("Capacity exceeding for nodes " + str(each_route) + " and " + str(edge_j))
						break
			elif result_dict[edge_i]==0 and result_dict[edge_j]==1:
				for each_route in result_list:
					if edge_j in each_route:
						length_of_list = len(each_route)
						total_demand = 0
						total_demand += demand_dict[edge_i]
						for each_cap in each_route:
							total_demand += demand_dict[each_cap]
						if total_demand <= vehicle_capacity:
								index_of_tuple_for_j = each_route.index(edge_j)
								if index_of_tuple_for_j == 0 or index_of_tuple_for_j == length_of_list-1:
									route = list(each_route)
									route.insert(length_of_list, edge_i)
									cycle_tuple  = tuple(route)
									result_dict[edge_i] = 1
									index_of_result_tuple = result_list.index(each_route)
									result_list[index_of_result_tuple] = cycle_tuple
								else:
									print ("Can't add the "+ str(edge_i) +" node as " + str(edge_j) +" is intermediate node")
						else:
							print ("Capacity exceeding for nodes " + str(each_route) + " and " + str(edge_i))
						break
			elif result_dict[edge_i]==0 and result_dict[edge_j]==0:
				route = []
				route.append(edge_i)
				route.append(edge_j)
				cycle_tuple  = tuple(route)
				result_dict[edge_i] = 1
				result_dict[edge_j] = 1
				result_list.append(cycle_tuple)
				print (result_list)
				print (result_dict)
			else:
				print ("Nodes exist in list" + " for " + str(edge_i) + "," + str(edge_j))
		else:
			print ("Iteration No.: " + str(i) + " for " + str(edge_i) + "," + str(edge_j))
			capacity_temp = demand_dict[edge_i] + demand_dict[edge_j]
			print ("Capacity Constraints Violated " + str(capacity_temp))
		###############################################################################
	else:
		if demand_dict[edge_i] + demand_dict[edge_j] <= vehicle_capacity :
			print ("Iteration No.: " + str(i) + " for " + str(edge_i) + "," + str(edge_j))
			print ("Capacity Constraints Not Violated in else")
			route = []
			route.append(edge_i)
			route.append(edge_j)
			cycle_tuple  = tuple(route)
			result_list.append(cycle_tuple)
			result_dict[edge_i] = 1
			result_dict[edge_j] = 1
			print (result_list)
			print (result_dict)

print ("********************* Final Results *********************")
for each_tuple in result_list:
	temp_list = list(each_tuple)
	temp_list.insert(0,0)
	temp_list.insert(len(temp_list),0)
	cycle_tuple = tuple(temp_list)
	result_list[result_list.index(each_tuple)] = cycle_tuple

for key,value in result_dict.items():
	if value == 0:
		result_list.append(tuple([0,key,0]))

for item in result_list:
	edge_list = list(item)
	G.add_nodes_from(edge_list)
	for i in range(len(edge_list)-1):
		edge_to_add_in_graph = (edge_list[i],edge_list[i+1])
		G.add_edge(*edge_to_add_in_graph)
print (G.nodes())
nx.draw_networkx(G,arrows=True,node_color = 'g', alpha = 0.8)
plt.savefig("simple_path.png") # save as png
plt.show() # display

print (result_list)
print (result_dict)
