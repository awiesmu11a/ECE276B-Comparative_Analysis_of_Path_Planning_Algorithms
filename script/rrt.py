import numpy as np
import math

class RRTNode(object):
    def __init__(self, key, coord):
        self.coord = coord
        self.key = key
        self.parent_node = None
        self.parent_action = None

class RRTEnvironment(object):
    def __init__(self, boundary, blocks, goal_coord, sampling, min_path_len, voxel_map, resolution = 0.1):
        self.boundary = boundary
        self.blocks = blocks
        self.goal_coord = goal_coord
        self.resolution = resolution
        self.dr = min_path_len
        self.map = voxel_map
        self.sample = sampling
    
    def getSample(self):
        boundary = self.boundary
        if self.sample == "uniform":
            return np.random.rand(3) * (boundary[0,3:6] - boundary[0,0:3]) + boundary[0,0:3]
        if self.sample == "biased":
            p = 0.4 # probability of sampling goal; can change for testing
            temp = np.random.rand(1)
            if temp < p:
                return self.goal_coord
            else:
                return np.random.rand(3) * (boundary[0,3:6] - boundary[0,0:3]) + boundary[0,0:3]


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
    

class RRTPlanner(object):
    @staticmethod
    def plan(start_coord, goal_coord, env, max_iter = 10000000, epsilon = 0.5):

        boundary = env.boundary
        blocks = env.blocks
        voxel = env.map
        resolution = env.resolution
        start_key = tuple(np.floor((start_coord - boundary[0,0:3]) / resolution))
        curr = RRTNode(start_key, start_coord)
        goal_key = tuple(np.floor((goal_coord - boundary[0,0:3]) / resolution))
        curr_key = start_key
        Nodes = {}
        Nodes[start_key] = curr
        Graph = {}
        
        for i in range(max_iter):
            if curr_key == goal_key:
                break
            rand_coord = env.getSample()

            # Find nearest node
            nearest_node = None
            nearest_key = None
            min_dist = np.inf
            for key, node in Nodes.items():
                dist = np.sqrt(sum((node.coord - rand_coord)**2))
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node
                    nearest_key = key
            
            # Find new node
            new_coord = nearest_node.coord + epsilon * (rand_coord - nearest_node.coord) / min_dist
            new_key = tuple(np.floor((new_coord - boundary[0,0:3]) / resolution))
            if np.sqrt(sum((new_coord - goal_coord)**2)) <= epsilon:
                new_coord = goal_coord
                new_key = goal_key
            if new_key in Nodes:
                continue
            if new_coord[0] < boundary[0,0] or new_coord[0] > boundary[0,3] or \
                new_coord[1] < boundary[0,1] or new_coord[1] > boundary[0,4] or \
                new_coord[2] < boundary[0,2] or new_coord[2] > boundary[0,5]:
                continue
            valid = True
            for k in range(blocks.shape[0]):
                if( new_coord[0] >= blocks[k,0] and new_coord[0] <= blocks[k,3] and\
                    new_coord[1] >= blocks[k,1] and new_coord[1] <= blocks[k,4] and\
                    new_coord[2] >= blocks[k,2] and new_coord[2] <= blocks[k,5] ):
                    valid = False
                    break
            if not valid:
                continue
            new_node = RRTNode(new_key, new_coord)
            if env.collision_check(nearest_node.coord, new_node.coord):
                continue

            Nodes[new_key] = new_node
            curr_key = new_key
            curr = new_node
            curr.parent_node = nearest_key
            if nearest_key not in Graph:
                Graph[nearest_key] = set()
            Graph[nearest_key].add(new_key)
            if new_key not in Graph:
                Graph[new_key] = set()
            Graph[new_key].add(nearest_key)

        if curr_key != goal_key:
            print("No path found!")
            return None, None, None
        path = []
        while curr_key != start_key:
            path.append(Nodes[curr_key].coord)
            curr_key = Nodes[curr_key].parent_node

        path.append(start_coord)
        path.reverse()
        path = np.array(path)
        pathlength = 0
        for i in range(path.shape[0]-1):
            pathlength += np.sqrt(sum((path[i+1] - path[i])**2))
        print("Path length for RRT: ", pathlength)
        return path, Nodes, Graph

