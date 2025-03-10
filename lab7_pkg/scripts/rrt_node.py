"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid
from bitarray import bitarray

# TODO: import as you need

# class def for tree nodes
# It's up to you if you want to use this
class Node(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.parent = None
        self.cost = None # only used in RRT*
        self.is_root = False


# Create a class for occupancy grid - this will be a 2d array of 0s and 1s.
# 0s represent free space, 1s represent occupied space.
# To store it, we have each row (y - coordinate) as a bit array.
# This way, we can access the occupancy grid as occupancy_grid[i][j].
# There will also be a method to convert from (x, y) coordinates to (i, j) indices.
class OccupancyGrid(object):
    def __init__(self, x_bounds, y_bounds, cell_size):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.cell_size = cell_size
        
        # The arrays will be a circular buffer of bitarrays
        self.start_row_index = 0
        
        # Create an array of bitarrays
        self.occupancy_grid = []
        num_arrays = int((x_bounds[1] - x_bounds[0]) / cell_size)
        length = int((y_bounds[1] - y_bounds[0]) / cell_size)
        for i in range(num_arrays):
            self.occupancy_grid.append(bitarray(length))
            self.occupancy_grid[i].setall(0)

    def __getitem__(self, coordinate):
        x, y = coordinate
        i = int((x - self.x_bounds[0]) / self.cell_size)
        i = (i + self.start_row_index) % len(self.occupancy_grid) # circular buffer
        j = int((y - self.y_bounds[0]) / self.cell_size)
        return self.occupancy_grid[i][j]
    
    def add_row(self, row):
        """
        Append a new row to the occupancy grid.
        """
        # Rewrite the (start_row_index)th row, and increment the start_row_index,
        # as this is a circular buffer.
        self.occupancy_grid[self.start_row_index] = row
        self.start_row_index = (self.start_row_index + 1) % len(self.occupancy_grid)


# class def for RRT
class RRT(Node):
    def __init__(self):
        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file
        pose_topic = "ego_racecar/odom"
        scan_topic = "/scan"
        self.declare_parameter('lookahead', 5.0)
        self.declare_parameter('max_distance', 2.0)
        self.declare_parameter('cell_size', 0.1)
        self.declare_parameter('goal_close_enough', 0.05)
        self.lookahead = self.get_parameter('lookahead').value
        self.max_distance = self.get_parameter('max_distance').value
        self.cell_size = self.get_parameter('cell_size').value
        self.goal_close_enough = self.get_parameter('goal_close_enough').value

        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.

        # TODO: create subscribers
        self.pose_sub_ = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.pose_callback,
            1)

        self.scan_sub_ = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            1)

        # publishers
        # TODO: create a drive message publisher, and other publishers that you might need

        # class attributes
        # TODO: maybe create your occupancy grid here
        self.occupancy_grid = OccupancyGrid((0, self.lookahead), (-self.lookahead, self.lookahead), self.cell_size)
        

    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        # for first time step, update the occupancy grid with the scan
        # for subsequent time steps, update the last row of the occupancy grid
        # with the scan


    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """

        return None

    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """

        x_bounds = (0, self.lookahead)
        x = np.random.uniform(x_bounds[0], x_bounds[1])
        y = np.random.uniform(-1*x, x)
        return (x, y)

    def nearest(self, tree, sampled_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """
        idx = np.argmin([LA.norm(np.array([node.x, node.y]) - np.array(sampled_point)) for node in tree])
        nearest_node = tree[idx]
        return nearest_node

    def steer(self, nearest_node, sampled_point):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """
        new_node = None
        # vector from nearest to sampled
        vec_to_sampled = np.array(sampled_point) - np.array([nearest_node.x, nearest_node.y])
        dist_to_sampled = LA.norm(vec_to_sampled)
        vec_to_sampled = vec_to_sampled / dist_to_sampled
        # if the distance to the sampled point is greater than the max distance
        # then we should only move max distance in the direction of the sampled point
        if dist_to_sampled > self.max_distance:
            new_node = Node()
            new_node.x = nearest_node.x + self.max_distance * vec_to_sampled[0]
            new_node.y = nearest_node.y + self.max_distance * vec_to_sampled[1]
            new_node.parent = nearest_node
        else:
            new_node = Node()
            new_node.x = sampled_point[0]
            new_node.y = sampled_point[1]
            new_node.parent = nearest_node

        return new_node

    def check_collision(self, nearest_node, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest (Node): nearest node on the tree
            new_node (Node): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """
        # check if the line between nearest and new_node is in collision
        node_dist = LA.norm(np.array([nearest_node.x, nearest_node.y]) - np.array([new_node.x, new_node.y]))
        cos_theta = (new_node.x - nearest_node.x) / node_dist
        sin_theta = (new_node.y - nearest_node.y) / node_dist
        num_points = int(node_dist / self.cell_size)
        i = np.arange(num_points)
        xs = nearest_node.x + i * self.cell_size * cos_theta
        ys = nearest_node.y + i * self.cell_size * sin_theta
        if np.any(self.occupancy_grid[xs, ys]):
            return True

        return False

    def is_goal(self, latest_added_node, goal_x, goal_y):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enough to the goal
        """
        dist_to_goal = LA.norm(np.array([latest_added_node.x, latest_added_node.y]) - np.array([goal_x, goal_y]))
        if dist_to_goal <= self.goal_close_enough:
            return True
        return False


    def find_path(self, tree, latest_added_node):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        path = []
        current_node = latest_added_node
        while current_node is not None:
            path.append(current_node)
            current_node = current_node.parent
        path.reverse()
        return path


    # The following methods are needed for RRT* and not RRT
    def cost(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        return 0

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (Node): node at one end of the straight line
            n2 (Node): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """
        return 0

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
        """
        neighborhood = []
        return neighborhood

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()