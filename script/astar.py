
# priority queue for OPEN list
from pqdict import pqdict
import numpy as np
import math

class AStarNode(object):
  def __init__(self, pqkey, coord, hval):
    self.pqkey = pqkey
    self.coord = coord
    self.g = math.inf
    self.h = hval
    self.parent_node = None
    self.parent_action = None
    self.closed = False
  def __lt__(self, other):
    return self.g < other.g   

class AStarEnvironment(object):
  def __init__(self, boundary, blocks, goal_coord, heuristic, min_path_len, voxel_map, resolution = 0.1):
    self.boundary = boundary
    self.blocks = blocks
    self.goal_coord = goal_coord
    self.resolution = resolution
    self.dr = min_path_len
    self.map = voxel_map
    self.heuristic = heuristic

  def getHeuristic(self, coord):
    if self.heuristic == "euclidean":
      return np.sqrt(sum((coord - self.goal_coord)**2))
    elif self.heuristic == "manhattan":
      return np.sum(np.abs(coord - self.goal_coord))
    elif self.heuristic == "chebyshev":
      return np.max(np.abs(coord - self.goal_coord))
    elif self.heuristic == "octile":
      return np.max(np.abs(coord - self.goal_coord)) + (np.sqrt(3) - 1) * np.min(np.abs(coord - self.goal_coord))
  
  def getCost(self, coord1, coord2):
    return np.sqrt(sum((coord1 - coord2)**2))
  
  def collision_check(self, A, B):
    length = np.sqrt(np.sum((A - B)**2))
    for i in range(self.blocks.shape[0]):
      for j in range(6):
        if j < 3:
          if A[j] == B[j]:
            continue
          t = (self.blocks[i,j] - A[j]) / (B[j] - A[j])
          x = A[0] + t * (B[0] - A[0])
          y = A[1] + t * (B[1] - A[1])
          z = A[2] + t * (B[2] - A[2])
          intersect = np.array([x,y,z])
          if intersect[0] >= self.blocks[i,0] and intersect[0] <= self.blocks[i,3] and\
              intersect[1] >= self.blocks[i,1] and intersect[1] <= self.blocks[i,4] and\
              intersect[2] >= self.blocks[i,2] and intersect[2] <= self.blocks[i,5] and \
              t >= 0 and t <= 1 and np.sqrt(np.sum((intersect - A)**2)) < length:
            return True
        else:
          if A[j-3] == B[j-3]:
            continue
          t = (self.blocks[i,j] - A[j-3]) / (B[j-3] - A[j-3])
          x = A[0] + t * (B[0] - A[0])
          y = A[1] + t * (B[1] - A[1])
          z = A[2] + t * (B[2] - A[2])
          intersect = np.array([x,y,z])
          if intersect[0] >= self.blocks[i,0] and intersect[0] <= self.blocks[i,3] and\
              intersect[1] >= self.blocks[i,1] and intersect[1] <= self.blocks[i,4] and\
              intersect[2] >= self.blocks[i,2] and intersect[2] <= self.blocks[i,5] and \
              t >= 0 and t <= 1 and np.sqrt(np.sum((intersect - A)**2)) < length:
            return True
    return False
  
  def getNeighbours(self, coord):
    numofdirs = 26
    [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
    dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
    dR = np.delete(dR,13,axis=1)
    dR = (dR / np.sqrt(np.sum(dR**2,axis=0))) * self.dr
    neighbours = []
    for k in range(numofdirs):
      next = coord + dR[:,k]
      if self.collision_check(coord, next):
        continue
      if( next[0] < self.boundary[0,0] or next[0] > self.boundary[0,3] or \
          next[1] < self.boundary[0,1] or next[1] > self.boundary[0,4] or \
          next[2] < self.boundary[0,2] or next[2] > self.boundary[0,5] ):
        continue
      valid = True
      for k in range(self.blocks.shape[0]):
        if( next[0] >= self.blocks[k,0] and next[0] <= self.blocks[k,3] and\
            next[1] >= self.blocks[k,1] and next[1] <= self.blocks[k,4] and\
            next[2] >= self.blocks[k,2] and next[2] <= self.blocks[k,5] ):
          valid = False
          break
      if not valid:
        continue
      if (np.sum((next - self.goal_coord)**2) <= 0.2):
        neighbours.append(self.goal_coord)
      else:
        neighbours.append(next)
    return neighbours

class AStar(object):
  @staticmethod
  def plan(start_coord, goal_coord, environment, epsilon = 1):
    print("Planning with A*")
    boundary = environment.boundary
    blocks = environment.blocks
    voxel = environment.map
    resolution = environment.resolution
    counter = 0
    # Initialize the graph and open list
    Graph = {}
    Nodes = {}
    OPEN = pqdict()
    CLOSED = set()
    
    # current node
    start_voxel = np.floor(start_coord / resolution)
    goal_voxel = np.floor(goal_coord / resolution)
    curr = AStarNode(tuple(start_voxel), start_coord, environment.getHeuristic(start_coord))
    curr.g = 0
    OPEN[curr.pqkey] = curr.g + epsilon * curr.h
    Nodes[curr.pqkey] = curr

    while not (tuple(goal_voxel) in CLOSED):

      curr = OPEN.popitem()
      curr_key = curr[0]
      curr_node = Nodes[curr_key]
      curr_node.closed = True
      CLOSED.add(curr_key)
      neighbours = environment.getNeighbours(curr_node.coord)

      for neighbour in neighbours:
        neighbour_key = tuple(np.floor(neighbour / resolution))
        if neighbour_key in CLOSED:
          continue
        if neighbour_key not in Nodes:
          neighbour_node = AStarNode(neighbour_key, neighbour, environment.getHeuristic(neighbour))
          neighbour_node.g = curr_node.g + environment.getCost(curr_node.coord, neighbour_node.coord)
          neighbour_node.parent_node = curr_key
          neighbour_node.parent_action = neighbour_node.coord - curr_node.coord                     # No use
          if Graph.get(curr_key) is None:
            Graph[curr_key] = set()
          Graph[curr_key].add(neighbour_key)
          Nodes[neighbour_key] = neighbour_node
          OPEN[neighbour_key] = neighbour_node.g + epsilon * neighbour_node.h
        else:
          neighbour_node = Nodes[neighbour_key]
        if neighbour_node.closed:
          continue
        if neighbour_node.g > curr_node.g + environment.getCost(curr_node.coord, neighbour_node.coord):
          neighbour_node.g = curr_node.g + environment.getCost(curr_node.coord, neighbour_node.coord)
          neighbour_node.parent_node = curr_key
          neighbour_node.parent_action = neighbour_node.coord - curr_node.coord                     # No use
          OPEN[neighbour_key] = neighbour_node.g + epsilon * neighbour_node.h
          if Graph.get(curr_key) is None:
            Graph[curr_key] = set()
          Graph[curr_key].add(neighbour_key)
      
    if tuple(goal_voxel) not in Nodes:
      print("No path found")
      return None, None, None
    
    # Backtrack to get the path
    path = []
    curr_key = tuple(goal_voxel)
    while curr_key != tuple(start_voxel):
      path.append(Nodes[curr_key].coord)
      curr_key = Nodes[curr_key].parent_node
    path.append(start_coord)
    path.reverse()

    pathlength = 0

    for i in range(len(path)-1):
      pathlength += np.sqrt(np.sum((path[i+1] - path[i])**2))
    print("Path length for A*: ", pathlength)

    return np.array(path), Graph, Nodes
  


    




    




