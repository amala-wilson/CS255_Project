from random import randrange

import copy
import random
import time
import numpy as np
import xml.etree.ElementTree as ET
import multiprocessing as mp

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
def two_opt(dist_matrix, city_tour):
  best_route = copy.deepcopy(city_tour)
  i, j = random.sample(range(0, len(city_tour[0])-1), 2)
  if i > j:
    i, j = j, i
  best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))
  best_route[0][-1] = best_route[0][0]
  best_route[1] = distance_calc(dist_matrix, best_route)

  return best_route

# ***************************************************
# nearest_neighbor
# ***************************************************
def nearest_neighbor(current_point, visited):
  min_val = 1000
  nn = 0
  for i in range(len(current_point)):
    distance = current_point[i]
    if distance == float(0):
      continue
    elif distance < min_val and visited[i] == "false":
      min_val = distance
      nn = i

  return nn, min_val

# ***************************************************
# nearest_neighbor_ktsp
# ***************************************************
def nearest_neighbor_ktsp(current_city, idx_location, dist_matrix, visited, k):
  city_count = 0
  tour = [[], float("inf")]
  distance = float(0)

  tour[0].append(idx_location)

  for t in range(len(current_city)-1):
    nn_sol = nearest_neighbor(current_city, visited)
    nn_location = nn_sol[0]
    distance += nn_sol[1]
    tour[0].append(nn_location)
    current_city = dist_matrix[nn_location]
    visited[nn_location] = "true"
    city_count += 1

    if city_count == k:
      break;

  tour[0].append(idx_location)
  distance += dist_matrix[nn_location][idx_location]
  tour[1] = distance

  return tour 

# ***************************************************
# generate_initial_solution
# ***************************************************
def generate_initial_solution(dist_matrix, visited, vertex_cnt, k):
  # Pick a random city to start the tour
  #random_city = randrange(0,vertex_cnt)
  random_city = 0
  #print("Random City: ",random_city)
  # Extract the distances to other cities from the chosen city
  starting_city = dist_matrix[random_city]
  #print("Starting City: ",starting_city)
  # Mark the starting city as visited
  visited[random_city] = "true"
  # Construct solution
  return nearest_neighbor_ktsp(starting_city, random_city, dist_matrix, visited, k)

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

def seed_function(dist_matrix, k):
  seed = [[],float("inf")]
  sequence = random.sample(list(range(1,dist_matrix.shape[0]+1)), dist_matrix.shape[0])
  #print("Sequence: ",sequence)
  i = 0
  k_cities_dict = {}  
  new_sequence = []
  while i != k:
    num = randrange(0, len(sequence))
    if k_cities_dict.get(num) == None:
      k_cities_dict[num] = "true"  
      new_sequence.append(num)
    i += 1

  #print("k Sequence: ",new_sequence)
  seed[0] = new_sequence
  seed[1] = distance_calc(dist_matrix, seed)
  return seed


# ***************************************************
# variable_neighborhood_search
# ***************************************************
def variable_neighborhood_search(dist_matrix, visited, iterations, max_attempts, neighborhood_size, vertex_cnt, k):
  count = 0

  initial_solution = generate_initial_solution(dist_matrix, visited, vertex_cnt, k)
  #initial_solution = seed_function(dist_matrix, k) 
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
if __name__ == '__main__':
  # Prepare dataset
  filename = "att48.xml"
  distance_matrix = parseXML(filename)
  #print("Distance Matrix: ", distance_matrix)
  distance_matrix_cp = np.array(distance_matrix)
  #print("Distance Matrix Cp: ", distance_matrix_cp)

  count = 0
  visited = []
  for i in distance_matrix_cp:
    visited.insert(count, "false")
    count += 1

  #print("Length of visited array: ", len(visited))
  #for i in visited:
  #  print("Visited value: ", i)

  # Variable Neighborhood Search
  k = count/4  # Subset of k cities of n cities total
  iterations = 10
  max_attempts = 20
  neighborhood_size = 2
  start = time.perf_counter()
  final_sol = variable_neighborhood_search(distance_matrix_cp, visited, iterations, max_attempts, neighborhood_size, count, k)
  print("Solution: ",final_sol)
  finish = time.perf_counter()
  print("Total time: ",finish-start)
