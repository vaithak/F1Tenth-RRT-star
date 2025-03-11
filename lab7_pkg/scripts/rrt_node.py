#!/usr/bin/env python3

"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
from scipy import ndimage
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
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion


SIMULATION = True
DEBUG = True

# class def for tree nodes
# It's up to you if you want to use this
class TreeNode(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.parent = None
        self.cost = None # only used in RRT*
        self.is_root = False


# Create a class for occupancy grid - this will be a 2d array of 0s and 1s.
# 0s represent free space, 1s represent occupied space.
# This way, we can access the occupancy grid as occupancy_grid[i][j].
# There will also be a method to convert from (x, y) coordinates to (i, j) indices.
class OccupancyGridManager:
    def __init__(self, x_bounds, y_bounds, cell_size, obstacle_inflation_radius, publisher):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.cell_size = cell_size
        self.publisher = publisher
        self.obstacle_inflation_radius = obstacle_inflation_radius
        
        # Initialize the occupancy grid.
        self.occupancy_grid = []
        num_arrays = int((x_bounds[1] - x_bounds[0]) / cell_size)
        length = int((y_bounds[1] - y_bounds[0]) / cell_size)
        for i in range(num_arrays):
            self.occupancy_grid.append([])
            for j in range(length):
                self.occupancy_grid[i].append(0)
        self.occupancy_grid = np.array(self.occupancy_grid, dtype=np.int8)
        
    def __getitem__(self, coordinate):
        x, y = coordinate
        i = np.int32((x - self.x_bounds[0]) / self.cell_size)
        j = np.int32((y - self.y_bounds[0]) / self.cell_size)
        return self.occupancy_grid[i, j]

    def __setitem__(self, coordinate, value):
        x, y = coordinate
        i = int((x - self.x_bounds[0]) / self.cell_size)
        j = int((y - self.y_bounds[0]) / self.cell_size)
        self.occupancy_grid[i, j] = value

    def populate(self, scan_msg):
        """
        Populate the occupancy grid based on the laser scan data.
        """
        # Get the ranges from the scan message
        ranges = scan_msg.ranges
        angle_increment = scan_msg.angle_increment

        start_angle, end_angle = np.radians(-90), np.radians(90)
        start_idx = (start_angle - scan_msg.angle_min) / angle_increment
        end_idx = (end_angle - scan_msg.angle_min) / angle_increment

        # Make the whole occupancy grid 0.
        np.putmask(self.occupancy_grid, self.occupancy_grid == 1, 0)

        for idx in range(int(start_idx), int(end_idx)):
            angle = scan_msg.angle_min + idx * angle_increment
            x = ranges[idx] * np.cos(angle)
            y = ranges[idx] * np.sin(angle)
            if np.isinf(x) or np.isinf(y) or np.isnan(x) or np.isnan(y):
                continue
            if x < self.x_bounds[0] or x >= self.x_bounds[1] or y < self.y_bounds[0] or y >= self.y_bounds[1]:
                continue
            # Cells along the line from the origin to the point are already set to 0.
            # Set the cell at the point to 1.
            self[x, y] = 1

        # Inflate the obstacles by the inflation radius using numpy's operations.
        inflation_radius = int(self.obstacle_inflation_radius / self.cell_size)
        mask_struct = ndimage.generate_binary_structure(2, 2)
        inflated_grid = ndimage.binary_dilation(self.occupancy_grid, mask_struct, iterations=inflation_radius)
        self.occupancy_grid = inflated_grid.astype(np.int8)

    def publish_for_vis(self):
        """
        Publish the occupancy grid for visualization.
        """
        msg = OccupancyGrid()
        msg.header.frame_id = "ego_racecar/laser"
        msg.info.width = len(self.occupancy_grid)
        msg.info.height = len(self.occupancy_grid[0])
        msg.info.resolution = self.cell_size
        msg.info.origin.position.x = self.x_bounds[0]
        msg.info.origin.position.y = self.y_bounds[0]
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = (100 * self.occupancy_grid.T).flatten().tolist()
        self.publisher.publish(msg)


# class def for RRT
class RRT(Node):
    def __init__(self):
        # Topics
        scan_topic = "/scan"
        if SIMULATION:
            pose_topic = "/ego_racecar/odom"
        else:
            pose_topic = "/pf/pose/odom"

        super().__init__('rrt_node')
        self.get_logger().info("RRT Node has been initialized")
        
        self.declare_parameter('lookahead', 5.0)
        self.declare_parameter('max_distance', 2.0)
        self.declare_parameter('cell_size', 0.05)
        self.declare_parameter('goal_bias', 0.05)
        self.declare_parameter('goal_close_enough', 0.05)
        self.declare_parameter('obstacle_inflation_radius', 0.20)
        self.declare_parameter('num_rrt_points', 200)
        self.declare_parameter('waypoint_file', '/home/vaithak/Downloads/UPenn/F1Tenth/sim_ws/src/sampling-based-motion-planning-team6/waypoints/fitted_waypoints.csv')

        self.lookahead = self.get_parameter('lookahead').value
        self.max_distance = self.get_parameter('max_distance').value
        self.cell_size = self.get_parameter('cell_size').value
        self.goal_bias = self.get_parameter('goal_bias').value
        self.goal_close_enough = self.get_parameter('goal_close_enough').value
        self.obstacle_inflation_radius = self.get_parameter('obstacle_inflation_radius').value
        self.num_rrt_points = self.get_parameter('num_rrt_points').value
        waypoint_file = self.get_parameter('waypoint_file').value
        self.waypoints = np.genfromtxt(waypoint_file, delimiter=',')

        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.

        # Create subscribers
        self.pose_sub_ = self.create_subscription(
            Odometry,
            pose_topic,
            self.pose_callback,
            1)

        self.scan_sub_ = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            1)

        # Publishers
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.goal_vis_pub_ = self.create_publisher(Marker, '/rrt/goal', 10)
        self.grid_vis_pub_ = self.create_publisher(OccupancyGrid, '/rrt/grid', 10)
        self.path_vis_pub_ = self.create_publisher(Marker, '/rrt/path', 10)
        self.tree_vis_pub_ = self.create_publisher(MarkerArray, '/rrt/tree', 10)
        self.waypoint_to_track_pub_ = self.create_publisher(PointStamped, '/waypoint_to_track', 10)

        # Create an occupancy grid of nav2_msgs/OccupancyGrid type
        self.grid_bounds_x = (0.0, self.lookahead)
        self.grid_bounds_y = (-self.lookahead/2, self.lookahead/2)
        self.occupancy_grid = OccupancyGridManager(self.grid_bounds_x,
                                                   self.grid_bounds_y,
                                                   self.cell_size, 
                                                   self.obstacle_inflation_radius, 
                                                   self.grid_vis_pub_)

        # Create a list to store the tree nodes
        self.tree = []


    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        self.occupancy_grid.populate(scan_msg)
        if DEBUG:
            self.occupancy_grid.publish_for_vis()


    def transform_point_to_car_frame(self, point, car_pose):
        # Transform the point to the car's frame
        x = point[0] - car_pose.position.x
        y = point[1] - car_pose.position.y
        # Rotate the point to the car's frame
        yaw = euler_from_quaternion([car_pose.orientation.x,
                                       car_pose.orientation.y,
                                       car_pose.orientation.z,
                                       car_pose.orientation.w])[2]
        x_car = x * np.cos(yaw) + y * np.sin(yaw)
        y_car = -x * np.sin(yaw) + y * np.cos(yaw)
        return np.array([x_car, y_car])


    def visualize_goal(self, goal_point):
        # Visualize the goal point
        marker = Marker()
        marker.header.frame_id = "ego_racecar/laser"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color.a = 1.0
        marker.color.g = 1.0
        marker.points.append(Point(x=goal_point[0], y=goal_point[1], z=0.0))
        self.goal_vis_pub_.publish(marker)


    def visualize_tree(self):
        """
        Mark nodes as purple points, and lines between nodes as blue lines.
        """
        marker_array = MarkerArray()

        marker = Marker()
        marker.header.frame_id = "ego_racecar/laser"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color.a = 1.0
        marker.color.r = 0.5
        marker.color.b = 0.5
        marker.id = 0
        for node in self.tree:
            marker.points.append(Point(x=node.x, y=node.y, z=0.0))
        marker_array.markers.append(marker)

        marker = Marker()
        marker.header.frame_id = "ego_racecar/laser"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color.a = 0.6
        marker.color.b = 1.0
        marker.id = 1
        for node in self.tree:
            if node.parent is not None:
                marker.points.append(Point(x=node.x, y=node.y, z=0.0))
                marker.points.append(Point(x=node.parent.x, y=node.parent.y, z=0.0))
        marker_array.markers.append(marker)

        self.tree_vis_pub_.publish(marker_array)


    def visualize_path(self, path):
        # Visualize the path
        marker = Marker()
        marker.header.frame_id = "ego_racecar/laser"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        for node in path:
            marker.points.append(Point(x=node.x, y=node.y, z=0.0))
        self.path_vis_pub_.publish(marker)


    def publish_waypoint_to_track(self, waypoint):
        # Publish the waypoint to track
        point = PointStamped()
        point.header.frame_id = "ego_racecar/laser"
        point.header.stamp = self.get_clock().now().to_msg()
        point.point.x = waypoint[0]
        point.point.y = waypoint[1]
        self.waypoint_to_track_pub_.publish(point)


    def rrt(self, goal):
        """
        The main RRT function
        Args: 
            goal (Pose): goal pose in car frame
        Returns:
            path ([]): list of points representing the path
        """
        # Initialize the tree with the start node
        start_node = TreeNode()
        start_node.x = 0.0 # car_pose.position.x in car frame
        start_node.y = 0.0 # car_pose.position.y
        start_node.is_root = True
        self.tree = [start_node]

        # Initialize the path with the start node
        path = [start_node]

        # Initialize the latest added node with the start node

        # RRT loop
        for _ in range(self.num_rrt_points):
            # Sample a point in the free space
            sampled_point = self.sample(goal)

            # Find the nearest node on the tree to the sampled point
            nearest_node = self.nearest(self.tree, sampled_point)

            # Steer from the nearest node to the sampled point
            new_node = self.steer(nearest_node, sampled_point)

            # Check if the path between the nearest node and the new node is in collision
            if self.check_collision(nearest_node, new_node):
                continue

            # Add the new node to the tree
            self.tree.append(new_node)

            # Check if the new node is close enough to the goal
            if self.is_goal(new_node, goal[0], goal[1]):
                path = self.find_path(self.tree, new_node)
                break

        return path


    def pose_callback(self, pose_msg):
        """
        The pose callback when subscrdibed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        """
        # Get the current x, y position of the vehicle
        pose = pose_msg.pose.pose
        x = pose.position.x
        y = pose.position.y

        # Get the current goal
        closest_index = np.argmin(np.linalg.norm(self.waypoints[:, :2] - np.array([x, y]), axis=1))
        for i in range(closest_index, closest_index + len(self.waypoints)):
            if i >= len(self.waypoints):
                i = i - len(self.waypoints)
            curr_dist = np.linalg.norm(self.waypoints[i, :2] - np.array([x, y]))
            if curr_dist > 0.9 * self.lookahead:
                goal_point = self.waypoints[i, :2]
                break

        goal_point_car_frame = self.transform_point_to_car_frame(goal_point, pose)
        # Visualize the goal point
        if DEBUG:
            self.visualize_goal(goal_point_car_frame)

        # Find the path from the current pose to the goal
        path = self.rrt(goal_point_car_frame)
        if path is None:
            return None

        # Visualize the path
        if DEBUG:
            self.visualize_path(path)
            self.visualize_tree()

        # Find the waypoint to track
        waypoint_to_track = path[0]

        # Publish the waypoint to track
        self.publish_waypoint_to_track((waypoint_to_track.x, waypoint_to_track.y))


    def sample(self, goal):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """

        # Goal biasing
        if np.random.uniform() < self.goal_bias:
            return (goal[0], goal[1])

        x_bounds = self.grid_bounds_x
        x = np.random.uniform(x_bounds[0], x_bounds[1])
        # Sample y from -x to x, but also within the grid bounds for y
        y_bounds = np.max([-x, self.grid_bounds_y[0]]), np.min([x, self.grid_bounds_y[1]])
        y = np.random.uniform(y_bounds[0], y_bounds[1])
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
            new_node = TreeNode()
            new_node.x = nearest_node.x + self.max_distance * vec_to_sampled[0]
            new_node.y = nearest_node.y + self.max_distance * vec_to_sampled[1]
            new_node.parent = nearest_node
        else:
            new_node = TreeNode()
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