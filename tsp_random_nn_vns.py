from random import randrange

import copy
import random
import time
import numpy as np
import xml.etree.ElementTree as ET


# ***************************************************
# parseXML
# ***************************************************
def parseXML(xmlfile):
  distance_matrix = []

  # Create element tree object
  tree = ET.parse(xmlfile)

  # Get root element
  root = tree.getroot()

  vertex_count = 0
  for x in root[5]:
    vertex = []
    vertex_count += 1
    vertex.insert(vertex_count, 0.0)
    for y in x:
      vertex.insert(int(y.text), y.attrib['cost'])
    distance_matrix.append(vertex)

  return distance_matrix

# ***************************************************
# distance_calc
# ***************************************************
def distance_calc(city_tour):
  distance = 0
  for k in range(0, len(city_tour[0])-1):
    m = k + 1
    distance = distance + np.float(distance_matrix_cp[city_tour[0][k]-1, city_tour[0][m]-1])
  
  return distance

# ***************************************************
# two_opt
# ***************************************************
def two_opt(dist_matrix, city_tour):
  best_route = copy.deepcopy(city_tour)
  i, j = random.sample(range(0, len(city_tour[0])-1), 2)
  if i > j:
    i, j = j, i
  best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))
  best_route[0][-1] = best_route[0][0]
  best_route[1] = distance_calc(best_route)

  return best_route

# ***************************************************
# random_nearest_neighbor
# ***************************************************
def random_nearest_neighbor(current_point, visited):
  
  while True:
    random_nn = randrange(0, len(current_point))
    if current_point[random_nn] == float(0):
      continue
    if visited[random_nn] != "true":
      break
  
  return random_nn, current_point[random_nn]

# ***************************************************
# random_nearest_neighbor_ktsp
# ***************************************************
def random_nearest_neighbor_ktsp(current_city, idx_location, dist_matrix, visited, k):
  city_count = 0
  tour = [[], float("inf")]
  distance = float(0)

  tour[0].append(idx_location)

  for t in range(len(current_city)-1):
    nn_sol = random_nearest_neighbor(current_city, visited)
    nn_location = nn_sol[0]
    distance += np.float(nn_sol[1])
    tour[0].append(nn_location)
    current_city = dist_matrix[nn_location]
    visited[nn_location] = "true"
    city_count += 1

    if city_count == k:
      break;

  tour[0].append(idx_location)
  distance += np.float(dist_matrix[nn_location][idx_location])
  tour[1] = distance

  return tour 

# ***************************************************
# generate_initial_solution
# ***************************************************
def generate_initial_solution(dist_matrix, visited, vertex_cnt, k):
  # Pick a random city to start the tour
  #random_city = randrange(0,vertex_cnt)
  #print("Random City: ",random_city)
  # Extract the distances to other cities from the chosen city
  starting_city = dist_matrix[0]
  # Mark the starting city as visited
  visited[0] = "true"
  # Construct solution
  return random_nearest_neighbor_ktsp(starting_city, 0, dist_matrix, visited, k)

# ***************************************************
# local_search
# ***************************************************
def local_search(dist_matrix, city_tour, max_attempts = 50, neighborhood_size = 2):
  count = 0
  solution = copy.deepcopy(city_tour)
  while (count < max_attempts):
    for i in range(0, neighborhood_size):
      candidate = two_opt(dist_matrix, city_tour = solution)
    if candidate[1] < solution[1]:
      solution = copy.deepcopy(candidate)
      count = 0
    else:
      count += 1

  return solution


# ***************************************************
# variable_neighborhood_search
# ***************************************************
def variable_neighborhood_search(dist_matrix, visited, iterations, max_attempts, neighborhood_size, vertex_cnt, k):
  count = 0

  initial_solution = generate_initial_solution(dist_matrix, visited, vertex_cnt, k);
  print("Initial Solution: ", initial_solution)
  solution = copy.deepcopy(initial_solution)
  best_solution = copy.deepcopy(initial_solution)
  while(count < iterations):
    for i in range(0, neighborhood_size):
      for j in range(0, neighborhood_size):
        solution = two_opt(dist_matrix, city_tour = best_solution)
        #print("2-opt Solution: ",solution)
      solution = local_search(dist_matrix, city_tour = solution, max_attempts = max_attempts, neighborhood_size = neighborhood_size)
      if solution[1] < best_solution[1]:
        best_solution = copy.deepcopy(solution)
        break
    count += 1
  return best_solution

# ***************************************************
# main
# ***************************************************

# Prepare dataset
filename = "att48.xml"
distance_matrix = parseXML(filename)
# print("Distance Matrix: ",distance_matrix)
distance_matrix_cp = np.array(distance_matrix)
# print("Distance Matrix Cp: ",distance_matrix_cp)

count = 0
visited = []
for i in distance_matrix:
  count += 1
  visited.insert(count, "false")

#print("Length of visited array: ", len(visited))

# *************************************************
# VARIABLE NEIGHBORHOOD SEARCH
# *************************************************
k = count/4  # Subset of k cities of n cities total
iterations = 10000
max_attempts = 20
neighborhood_size = 2
start = time.perf_counter()
final_sol = variable_neighborhood_search(distance_matrix_cp, visited, iterations, max_attempts, neighborhood_size, count, k)
print("Final Solution: ",final_sol)
finish = time.perf_counter()
print("Total time: ",finish-start)


# *************************************************
# ONLY TESTING NEAREST NEIGHBOR SOLUTION
# *************************************************
"""
count = 0
visit_count = 0
max_count = 5
visited = []
while count != max_count:
  for i in distance_matrix:
    visit_count += 1
    visited.insert(visit_count, "false")
  visit_count = 0
  sol = generate_initial_solution()
  print("Solution: ",sol)
  count += 1
"""
