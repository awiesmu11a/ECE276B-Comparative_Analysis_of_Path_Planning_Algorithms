import numpy as np
import time
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import Planner
from astar import *
from rrt import *

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def collision_check(path, boundary, blocks):
  
  for k in range(path.shape[0] - 1):

    path_length = np.sqrt(np.sum((path[k+1,:] - path[k,:])**2))

    for i in range(blocks.shape[0]):

      for j in range(6):
        if j < 3:
          if path[k, j] == path[k+1, j]:
            continue
          t = (blocks[i,j] - path[k,j])/(path[k+1,j] - path[k,j])
          x = t * (path[k+1,0] - path[k,0]) + path[k,0]
          y = t * (path[k+1,1] - path[k,1]) + path[k,1]
          z = t * (path[k+1,2] - path[k,2]) + path[k,2]
          intersect = np.array([x,y,z])
          if intersect[0] >= blocks[i,0] and intersect[0] <= blocks[i,3] and\
             intersect[1] >= blocks[i,1] and intersect[1] <= blocks[i,4] and\
             intersect[2] >= blocks[i,2] and intersect[2] <= blocks[i,5] and\
             t > 0 and t < 1 and np.sqrt(np.sum((intersect - path[k,:])**2)) < path_length:
            return True, intersect
        else:
          if path[k, j-3] == path[k+1, j-3]:
            continue
          t = (blocks[i, j] - path[k, j-3]) / (path[k+1, j-3] - path[k, j-3])
          x = t * (path[k+1, 0] - path[k, 0]) + path[k, 0]
          y = t * (path[k+1, 1] - path[k, 1]) + path[k, 1]
          z = t * (path[k+1, 2] - path[k, 2]) + path[k, 2]
          intersect = np.array([x, y, z])
          if intersect[0] >= blocks[i, 0] and intersect[0] <= blocks[i, 3] and \
             intersect[1] >= blocks[i, 1] and intersect[1] <= blocks[i, 4] and \
             intersect[2] >= blocks[i, 2] and intersect[2] <= blocks[i, 5] and \
             t > 0 and t < 1 and np.sqrt(np.sum((intersect - path[k, :]) ** 2)) < path_length:
            return True, intersect
        
  return False, np.zeros(3)

        
def voxel_gen(boundary, blocks):

  # First generate the voxels for the complete environment
  # The voxel size is 0.1m
  voxel_size = 0.1
  voxel_map = np.zeros((int(abs((boundary[0,3]-boundary[0,0])/voxel_size)),\
                        int(abs((boundary[0,4]-boundary[0,1])/voxel_size)),\
                        int(abs((boundary[0,5]-boundary[0,2])/voxel_size))),dtype=bool)
  
  # Fill in the voxels for the blocks
  for k in range(blocks.shape[0]):
    xmin = int((blocks[k,0]-boundary[0,0])/voxel_size)
    xmax = int((blocks[k,3]-boundary[0,0])/voxel_size)
    ymin = int((blocks[k,1]-boundary[0,1])/voxel_size)
    ymax = int((blocks[k,4]-boundary[0,1])/voxel_size)
    zmin = int((blocks[k,2]-boundary[0,2])/voxel_size)
    zmax = int((blocks[k,5]-boundary[0,2])/voxel_size)
    voxel_map[xmin:xmax,ymin:ymax,zmin:zmax] = True

  print(voxel_map.shape)

    # Plot the voxel map
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.voxels(voxel_map, edgecolor='k')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show(block=True)

  # Check if the path collides with the blocks using 3D variant of Bresenham's line algorithm
  return voxel_map


def load_map(fname):
  '''
  Loads the bounady and blocks from map file fname.
  
  boundary = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  
  blocks = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'],
            ...,
            ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  '''
  mapdata = np.loadtxt(fname,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),\
                                    'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
  blockIdx = mapdata['type'] == b'block'
  boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  return boundary, blocks


def draw_map(boundary, blocks, start, goal):
  '''
  Visualization of a planning problem with environment boundary, obstacle blocks, and start and goal points
  '''
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  hb = draw_block_list(ax,blocks)
  hs = ax.plot(start[0:1],start[1:2],start[2:],'ro',markersize=7,markeredgecolor='k')
  hg = ax.plot(goal[0:1],goal[1:2],goal[2:],'go',markersize=7,markeredgecolor='k')  
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim(boundary[0,0],boundary[0,3])
  ax.set_ylim(boundary[0,1],boundary[0,4])
  ax.set_zlim(boundary[0,2],boundary[0,5])
  ax.mouse_init()
  return fig, ax, hb, hs, hg

def draw_block_list(ax,blocks):
  '''
  Subroutine used by draw_map() to display the environment blocks
  '''
  v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype='float')
  f = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,2,3],[4,5,6,7]])
  clr = blocks[:,6:]/255
  n = blocks.shape[0]
  d = blocks[:,3:6] - blocks[:,:3] 
  vl = np.zeros((8*n,3))
  fl = np.zeros((6*n,4),dtype='int64')
  fcl = np.zeros((6*n,3))
  for k in range(n):
    vl[k*8:(k+1)*8,:] = v * d[k] + blocks[k,:3]
    fl[k*6:(k+1)*6,:] = f + k*8
    fcl[k*6:(k+1)*6,:] = clr[k,:]
  
  if type(ax) is Poly3DCollection:
    ax.set_verts(vl[fl])
  else:
    pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
    pc.set_facecolor(fcl)
    h = ax.add_collection3d(pc)
    return h


def runtest(mapfile, start, goal, planner, verbose = True):
  '''
  This function:
   * loads the provided mapfile
   * creates a motion planner
   * plans a path from start to goal
   * checks whether the path is collision free and reaches the goal
   * computes the path length as a sum of the Euclidean norm of the path segments
  '''

  boundary, blocks = load_map(mapfile) # Boundary and blocks are 2D arrays
  voxel_map = voxel_gen(boundary, blocks) # Voxel map is a 3D array with obstacles as 1s and free space as 0s

  min_seg_len = 0.5 # The minimum length to get neighbours
  resolution = 0.1 # Resolution of the voxel map (Mainly used as keys for neighbours)

  if planner == "astar":

    t0_astar = tic()
    # Details of choosing the heuristic can be found in the report
    heuristic = "manhattan" # Choose your heuristic here: euclidean, manhattan, octile, chebyshev
    environment = AStarEnvironment(boundary, blocks, goal, heuristic, min_seg_len, voxel_map, resolution)
    path, graph, nodes = AStar.plan(start, goal, environment)
    toc(t0_astar,"A*")
    print("Heuristic: ", heuristic)

    print("Number of nodes in the graph: ", len(nodes))
  
  if planner == "rrt":
    
    t0_rrt = tic()
    # Details of choosing the sampling strategy can be found in the report
    sampling = "uniform" # Choose your sampling method here: uniform, biased
    environment = RRTEnvironment(boundary, blocks, goal, sampling, min_seg_len, voxel_map, resolution)
    path, graph, nodes = RRTPlanner.plan(start, goal, environment)
    toc(t0_rrt,"RRT")
    print(" Sampling: ", sampling)

    print("Number of nodes in the graph: ", len(nodes))

  if planner == "baseline":

    MP = Planner.MyPlanner(boundary, blocks)
      
    t0_baseline = tic()
    path = MP.plan(start, goal)
    toc(t0_baseline,"Baseline")

    # Collision check for baseline planner; Same used in A* and RRT
    collision, point = collision_check(path, boundary, blocks)
    print("Collision: ", collision)
    if collision: print("Point of collision: ", point)

    if verbose:      
      fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)
      ax.plot(path[:,0],path[:,1],path[:,2],'r-') # Baseline planner

    goal_reached = sum((path[-1]-goal)**2) <= 0.1
    success = (not collision) and goal_reached
    pathlength = np.sum(np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1)))
    return success, pathlength

  
  # Display the environment
  if verbose:
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)
  # Plot the path
  if verbose:
    ax.plot(path[:,0],path[:,1],path[:,2],'r-')

  if len(path) > 0:
    success = True
  else:
    success = False
  
  pathlength = np.sum(np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1)))
  return success, pathlength


def test_single_cube(planner, verbose = True):
  print('Running single cube test...\n') 
  start = np.array([2.3, 2.3, 1.3])
  goal = np.array([7.0, 7.0, 5.5])
  success, pathlength = runtest('./maps/single_cube.txt', start, goal, planner, verbose)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')
  
  
def test_maze(planner, verbose = True):
  print('Running maze test...\n') 
  start = np.array([0.0, 0.0, 1.0])
  goal = np.array([12.0, 12.0, 5.0])
  success, pathlength = runtest('./maps/maze.txt', start, goal, planner, verbose)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')

    
def test_window(planner, verbose = True):
  print('Running window test...\n') 
  start = np.array([0.2, -4.9, 0.2])
  goal = np.array([6.0, 18.0, 3.0])
  success, pathlength = runtest('./maps/window.txt', start, goal, planner, verbose)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')

  
def test_tower(planner, verbose = True):
  print('Running tower test...\n') 
  start = np.array([2.5, 4.0, 0.5])
  goal = np.array([4.0, 2.5, 19.5])
  success, pathlength = runtest('./maps/tower.txt', start, goal, planner, verbose)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')

     
def test_flappy_bird(planner, verbose = True):
  print('Running flappy bird test...\n') 
  start = np.array([0.5, 2.5, 5.5])
  goal = np.array([19.0, 2.5, 5.5])
  success, pathlength = runtest('./maps/flappy_bird.txt', start, goal, planner, verbose)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength) 
  print('\n')

  
def test_room(planner, verbose = True):
  print('Running room test...\n') 
  start = np.array([1.0, 5.0, 1.5])
  goal = np.array([9.0, 7.0, 1.5])
  success, pathlength = runtest('./maps/room.txt', start, goal, planner, verbose)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')


def test_monza(planner, verbose = True):
  print('Running monza test...\n')
  start = np.array([0.5, 1.0, 4.9])
  goal = np.array([3.8, 1.0, 0.1])
  success, pathlength = runtest('./maps/monza.txt', start, goal, planner, verbose)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')


if __name__=="__main__":

  env = sys.argv[1]
  planner = sys.argv[2]

  if planner != 'baseline' and planner != 'astar' and planner != 'rrt':
    print('Invalid planner name')
    sys.exit()

  if env == 'single_cube':
    test_single_cube(planner)
  elif env == 'maze':
    test_maze(planner)
  elif env == 'window':
    test_window(planner)
  elif env == 'tower':
    test_tower(planner)
  elif env == 'flappy_bird':
    test_flappy_bird(planner)
  elif env == 'room':
    test_room(planner)
  elif env == 'monza':
    test_monza(planner)
  else:
    print('Invalid environment name')

  plt.show(block=True)