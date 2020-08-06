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
      vertex.insert(int(y.text), float(y.attrib['cost']))
    distance_matrix.append(vertex)

  return distance_matrix

# ***************************************************
# generate_neighborhood
# ***************************************************
def generate_neighborhood(num_neighborhood, dist_matrix, best_solution):
  neighborhood_structures = []

  for i in range(num_neighborhood):
    k, j = random.sample(range(0, len(best_solution[0])-1), 2)
    neighborhood_structures.append(two_opt(best_solution, k, j, dist_matrix))

  return neighborhood_structures

# ***************************************************
# seed_function
# ***************************************************
def seed_function(dist_matrix, k):
  seed = [[],float("inf")]
  sequence = random.sample(list(range(1,dist_matrix.shape[0]+1)), dist_matrix.shape[0])

  i = 0
  k_cities_dict = {}  
  new_sequence = []

  while i != k:
    num = randrange(0, len(sequence))
    if k_cities_dict.get(num) == None:
      k_cities_dict[num] = "true"  
      new_sequence.append(num)
    i += 1

  seed[0] = new_sequence
  seed[1] = distance_calc(dist_matrix, seed)

  return seed

# ***************************************************
# distance_calc
# ***************************************************
def distance_calc(dist_matrix, city_tour):
  distance = 0
  for k in range(0, len(city_tour[0])-1):
    m = k + 1
    distance = distance + dist_matrix[city_tour[0][k]-1, city_tour[0][m]-1]
  
  return distance

# ***************************************************
# two_opt
# ***************************************************
def two_opt(city_tour, i, j, dist_matrix):
  best_route = copy.deepcopy(city_tour)

  best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))
  best_route[0][-1] = best_route[0][0]
  best_route[1] = distance_calc(dist_matrix, best_route)

  return best_route

# ***************************************************
# local_search
# ***************************************************
def local_search(city_tour, neighborhood_struct, dist_matrix, max_attempts = 50, neighborhood_size = 2):
  count = 0
  solution = copy.deepcopy(city_tour)

  while (count < max_attempts):
    for i in range(0, len(neighborhood_struct)):
      k, j = random.sample(range(0, len(solution[0])-1), 2)
      candidate = two_opt(solution, k, j, dist_matrix)
      if candidate[1] < solution[1]:
        solution = copy.deepcopy(candidate)
        count = 0
      else:
        count += 1

  return solution

# ***************************************************
# shake
# ***************************************************
def shake(city_tour, neighborhood_struct, dist_matrix):
  best_route = copy.deepcopy(city_tour)

  for i in range(len(neighborhood_struct)):
    j = i + 1
    while j < len(neighborhood_struct):
      new_route = two_opt(city_tour, i, j, dist_matrix)
      if new_route[1] < best_route[1]:
        best_route = new_route
        break
      j += 1

  return best_route

# ***************************************************
# variable_neighborhood_descent
# ***************************************************
def variable_neighborhood_descent(curr_tour, dist_matrix, max_attempts = 50, neighborhood_size = 2):
  neighborhood_count = 0

  solution = copy.deepcopy(curr_tour)
  best_solution = copy.deepcopy(curr_tour)

  while True:
    while neighborhood_count < neighborhood_size:
      neighborhood_struct = curr_tour[0]
      solution = local_search(solution, neighborhood_struct, dist_matrix, max_attempts, neighborhood_size)
      if solution[1] < best_solution[1]:
        best_solution = copy.deepcopy(solution)
        neighborhood_count = 0
      else:
        neighborhood_count += 1
    if neighborhood_count == neighborhood_size:
      break
  
  return best_solution

# ***************************************************
# general_variable_neighborhood_search
# ***************************************************
def general_variable_neighborhood_search(dist_matrix, iterations, max_attempts, neighborhood_size, k):
  count = 0

  while count < iterations:
    initial_solution = seed_function(dist_matrix, k)
    solution = copy.deepcopy(initial_solution)
    best_solution = copy.deepcopy(initial_solution)
    neighborhood_structures = generate_neighborhood(neighborhood_size, dist_matrix, best_solution)
    neighborhood_count = 0
    while neighborhood_count < neighborhood_size:
      neighborhood_struct = neighborhood_structures[neighborhood_count]
      solution = shake(solution, neighborhood_struct, dist_matrix)
      solution = variable_neighborhood_descent(solution, dist_matrix, max_attempts, neighborhood_size) 
      if solution[1] < best_solution[1]:
        best_solution = copy.deepcopy(solution)
        neighborhood_count = 0
      else:
        neighborhood_count += 1
    count += 1 
  
  return best_solution

# ***************************************************
# main
# ***************************************************
if __name__ == '__main__':
  # Prepare dataset
  #filename = "../data/att48.xml"
  filename = "../data/kroA200.xml"
  #filename = "pcb3038.xml"
  distance_matrix = parseXML(filename)
  distance_matrix_cp = np.array(distance_matrix)

  count = len(distance_matrix_cp)

  # General Variable Neighborhood Search
  k = count/4  # Subset of k cities of n cities total
  iterations = 100
  max_attempts = 20
  neighborhood_size = 2

  start = time.perf_counter()
  final_sol = general_variable_neighborhood_search(distance_matrix_cp, iterations, max_attempts, neighborhood_size, k)
  print("Solution: ",final_sol)
  finish = time.perf_counter()
  print("Total time: ",finish-start)
