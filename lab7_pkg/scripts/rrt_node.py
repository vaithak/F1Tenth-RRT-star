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
INVALID_INDEX = -1

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
    def __init__(self, x_bounds, y_bounds, cell_size, obstacle_inflation_radius, publisher, laser_frame):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.cell_size = cell_size
        self.publisher = publisher
        self.obstacle_inflation_radius = obstacle_inflation_radius
        self.laser_frame = laser_frame
        
        # Initialize the occupancy grid.
        self.occupancy_grid = []
        num_arrays = int((x_bounds[1] - x_bounds[0]) / cell_size) + 1
        length = int((y_bounds[1] - y_bounds[0]) / cell_size) + 1
        for i in range(num_arrays):
            self.occupancy_grid.append([])
            for j in range(length):
                self.occupancy_grid[i].append(0)
        self.occupancy_grid = np.array(self.occupancy_grid, dtype=np.int8)

        self.origin = self.compute_index_from_coordinates(0, 0)

        # Compute angles and range maxes for these angles according to the grid.
        # For angles < angles_of_intersection[0], ray from origin to the point will intersect with the right wall.
        # For angles > angles_of_intersection[1], ray from origin to the point will intersect with the left wall.
        # For angles in between, ray from origin to the point will intersect with the top wall.
        self.angles_of_intersection = (
            np.arctan2(y_bounds[0], x_bounds[1]),
            np.arctan2(y_bounds[1], x_bounds[1])
        )        

    def compute_index_from_coordinates(self, x, y):
        # 0, 0 in grid coordinates is top left corner.
        # But in (x, y) coordinates, top left corner is (x_bounds[1], y_bounds[1]).
        # Bottom left corner is (x_bounds[0], y_bounds[0]).
        # x goes from bottom to top, y goes from right to left.
        i = np.int32((x - self.x_bounds[0]) / self.cell_size)
        i = len(self.occupancy_grid) - i - 1
        j = np.int32((y - self.y_bounds[0]) / self.cell_size)
        j = len(self.occupancy_grid[0]) - j - 1
        # if i, j is out of bounds, put None.
        # Check if i is a numpy array,
        if isinstance(i, np.ndarray):
            np.putmask(i, i < 0, INVALID_INDEX)
            np.putmask(i, i >= len(self.occupancy_grid), INVALID_INDEX)
            np.putmask(j, j < 0, INVALID_INDEX)
            np.putmask(j, j >= len(self.occupancy_grid[0]), INVALID_INDEX)
        else:
            if i < 0 or i >= len(self.occupancy_grid) or j < 0 or j >= len(self.occupancy_grid[0]):
                return INVALID_INDEX, INVALID_INDEX
        return i, j

    def __getitem__(self, coordinate):
        x, y = coordinate
        i, j = self.compute_index_from_coordinates(x, y)
        # Get value only where i, j is not None.
        res = np.ones_like(i)
        np.putmask(res, i is not INVALID_INDEX and j is not INVALID_INDEX, self.occupancy_grid[i, j])

    def __setitem__(self, coordinate, value):
        x, y = coordinate
        i, j = self.compute_index_from_coordinates(x, y)
        # Put value only where i, j is not None.
        # Filter out the indices where i, j is None and only set the values where i, j is not None.
        valid_indices = np.logical_and(i != INVALID_INDEX, j != INVALID_INDEX)
        self.occupancy_grid[i[valid_indices], j[valid_indices]] = value
        

    def bresenham(self, x1, y1, x2, y2):
        """
        Bresenham's line algorithm.
        Traverse, the grid along the line from (x0, y0) to (x1, y1).
        """
        dx = x2 - x1
        dy = y2 - y1

        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)

        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True

        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1

        # Calculate error
        error = dx // 2
        ystep = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx

        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()

        return points


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

        # Make the whole occupancy grid as 0s.
        np.putmask(self.occupancy_grid, self.occupancy_grid == 1, 0)
        # Vectorize the above loop.
        ranges = np.array(ranges)[int(start_idx):int(end_idx+1)]
        angles = np.arange(start_angle, end_angle, angle_increment)
        xs = np.multiply(ranges, np.cos(angles))
        ys = np.multiply(ranges, np.sin(angles))
        self[xs, ys] = 1

        # Inflate the obstacles by the inflation radius using numpy's operations.
        inflation_radius = int(self.obstacle_inflation_radius / self.cell_size)
        mask_struct = ndimage.generate_binary_structure(2, 2)
        inflated_grid = ndimage.binary_dilation(self.occupancy_grid, mask_struct, iterations=inflation_radius)
        self.occupancy_grid = inflated_grid.astype(np.int8)

    def check_line_collision(self, x1, y1, x2, y2):
        """
        Check if the line between (x1, y1) and (x2, y2) is in collision with the occupancy grid.
        """
        grid_x1, grid_y1 = self.compute_index_from_coordinates(x1, y1)
        grid_x2, grid_y2 = self.compute_index_from_coordinates(x2, y2)
        if grid_x1 is None or grid_y1 is None or grid_x2 is None or grid_y2 is None:
            return True

        points = self.bresenham(grid_x1, grid_y1, grid_x2, grid_y2)
        for point in points:
            i, j = point
            if self.occupancy_grid[i, j] == 1:
                return True
        return False

    def publish_for_vis(self):
        """
        Publish the occupancy grid for visualization.
        """
        msg = OccupancyGrid()
        msg.header.frame_id = self.laser_frame
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
        msg.data = np.fliplr(np.rot90(100 * self.occupancy_grid, k=1)).flatten().tolist()
        self.publisher.publish(msg)


# class def for RRT
class RRT(Node):
    def __init__(self):
        # Topics
        scan_topic = "/scan"
        self.laser_frame = "laser"
        if SIMULATION:
            pose_topic = "/ego_racecar/odom"
            self.laser_frame = "ego_racecar/laser"
        else:
            pose_topic = "/pf/pose/odom"
            self.laser_frame = "laser"

        super().__init__('rrt_node')
        self.get_logger().info("RRT Node has been initialized")
        
        self.declare_parameter('lookahead', 2.0)
        self.declare_parameter('max_steer_distance', 0.6)
        # self.declare_parameter('min_waypoint_tracking_distance', 0.6)
        # self.declare_parameter('waypoint_close_enough', 0.4)
        self.declare_parameter('rrt_delay_counter', 3)
        self.declare_parameter('cell_size', 0.1)
        self.declare_parameter('goal_bias', 0.1)
        self.declare_parameter('goal_close_enough', 0.05)
        self.declare_parameter('obstacle_inflation_radius', 0.20)
        self.declare_parameter('num_rrt_points', 100)
        self.declare_parameter('neighborhood_radius', 0.8) # Ensure this is greater than max_steer_distance
        self.declare_parameter('waypoint_file', '/home/vaithak/Downloads/UPenn/F1Tenth/sim_ws/src/sampling-based-motion-planning-team6/waypoints/fitted_waypoints.csv')

        self.lookahead = self.get_parameter('lookahead').value
        self.max_steer_distance = self.get_parameter('max_steer_distance').value
        # self.min_waypoint_tracking_distance = self.get_parameter('min_waypoint_tracking_distance').value
        # self.waypoint_close_enough = self.get_parameter('waypoint_close_enough').value
        # self.rrt_everytime = self.get_parameter('rrt_everytime').value
        self.rrt_delay_counter = self.get_parameter('rrt_delay_counter').value
        self.cell_size = self.get_parameter('cell_size').value
        self.goal_bias = self.get_parameter('goal_bias').value
        self.goal_close_enough = self.get_parameter('goal_close_enough').value
        self.obstacle_inflation_radius = self.get_parameter('obstacle_inflation_radius').value
        self.num_rrt_points = self.get_parameter('num_rrt_points').value
        self.neighborhood_radius = self.get_parameter('neighborhood_radius').value
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
        self.waypoint_to_track_pub_ = self.create_publisher(Marker, '/rrt/waypoint_to_track', 10)

        # Create an occupancy grid of nav2_msgs/OccupancyGrid type
        self.grid_bounds_x = (0.0, self.lookahead)
        self.grid_bounds_y = (-self.lookahead/2, self.lookahead/2)
        self.occupancy_grid = OccupancyGridManager(self.grid_bounds_x,
                                                   self.grid_bounds_y,
                                                   self.cell_size, 
                                                   self.obstacle_inflation_radius, 
                                                   self.grid_vis_pub_,
                                                   self.laser_frame)

        # RRT Variables
        self.tree = []
        self.current_following_waypoint = None
        self.grid_formed = False
        self.current_rrt_delay_counter = 0

        # Variables for pure pursuit
        self.prev_curvature = None
        self.kp_gain = 0.5
        self.kd_gain = 0.0
        self.adaptive_speed = lambda curvature: 4.0 / (1 + 2 * np.abs(curvature))


    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        self.occupancy_grid.populate(scan_msg)
        self.grid_formed = True
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
    

    def transform_point_to_map_frame(self, point, car_pose):
        # Transform the point to the map frame
        x = point[0]
        y = point[1]
        # Rotate the point to the map frame
        yaw = euler_from_quaternion([car_pose.orientation.x,
                                       car_pose.orientation.y,
                                       car_pose.orientation.z,
                                       car_pose.orientation.w])[2]
        x_map = x * np.cos(yaw) - y * np.sin(yaw) + car_pose.position.x
        y_map = x * np.sin(yaw) + y * np.cos(yaw) + car_pose.position.y
        return np.array([x_map, y_map])


    def visualize_goal(self, goal_point):
        # Visualize the goal point
        marker = Marker()
        marker.header.frame_id = self.laser_frame
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
        marker.header.frame_id = self.laser_frame
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
        marker.header.frame_id = self.laser_frame
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
        marker.header.frame_id = self.laser_frame
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


    def visualize_waypoint_to_track(self, waypoint):
        # Publish the waypoint to track
        marker = Marker()
        marker.header.frame_id = self.laser_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE 
        marker.action = Marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.position.x = waypoint[0]
        marker.pose.position.y = waypoint[1]
        marker.pose.position.z = 0.0
        self.waypoint_to_track_pub_.publish(marker)


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


    def rrt_star(self, goal):
        """
        The main RRT* function
        Args: 
            goal (Pose): goal pose in car frame
        Returns:
            path ([]): list of points representing the path
        """
        # Initialize the tree with the start node
        start_node = TreeNode()
        start_node.x = 0.0
        start_node.y = 0.0
        start_node.cost = 0
        start_node.is_root = True
        self.tree = [start_node]

        # Initialize the path with the start node
        path = [start_node]

        # RRT* loop
        goal_with_min_cost = None
        for _ in range(self.num_rrt_points):
            # Sample a point in the free space
            sampled_point = self.sample(goal)

            # Find the nearest node on the tree to the sampled point
            nearest_node = self.nearest(self.tree, sampled_point)

            # Steer from the nearest node to the sampled point
            new_node = self.steer(nearest_node, sampled_point)

            # Calculate the cost of the new node
            new_node.cost, new_node.parent = self.calc_cost_new_node(self.tree, new_node)
            if new_node.parent is None:
                continue

            self.tree.append(new_node)
            self.rewire(self.tree, new_node)

            # Check if the new node is close enough to the goal
            if self.is_goal(new_node, goal[0], goal[1]):
                if goal_with_min_cost is None or new_node.cost < goal_with_min_cost.cost:
                    goal_with_min_cost = new_node

        # Find the path from the closest node to the start node
        path = self.find_path(self.tree, goal_with_min_cost)
        return path


    def pure_pursuit(self, current_goal):
        """
        The pure pursuit controller
        Args: 
            current_goal (Pose): goal pose in car frame
        
        Publishes:
            AckermannDriveStamped: publishes the drive
        """
        # Calculate the curvature/steering angle
        dist_to_goal = np.sqrt(current_goal[0]**2 + current_goal[1]**2)
        if dist_to_goal < 1e-2:
            return
        curvature = 2 * current_goal[1] / (dist_to_goal**2)
        if self.prev_curvature is None:
            self.prev_curvature = curvature
        steering_angle = self.kp_gain * curvature + self.kd_gain * (curvature - self.prev_curvature)
        self.prev_curvature = curvature
        # if DEBUG:
            # self.get_logger().info(f'Steering angle: {steering_angle}')

        # Publish the drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'map'
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.adaptive_speed(curvature)
        # if DEBUG:
            # self.get_logger().info(f'Steering angle: {drive_msg.drive.steering_angle}')
            # self.get_logger().info(f'Speed: {drive_msg.drive.speed}')
        self.drive_pub_.publish(drive_msg)


    def calc_cost_path(self, path):
        """
        Calculate the cost of the path
        Args: 
            path ([]): list of points representing the path
        Returns:
            cost (float): the cost of the path
        """
        cost = 0
        for i in range(1, len(path)):
            cost += LA.norm(np.array([path[i].x, path[i].y]) - np.array([path[i-1].x, path[i-1].y]))
        return cost


    def find_waypoint_to_track(self, path):
        """
        Find the waypoint to track
        Args: 
            path ([]): list of points representing the path
        Returns:
            waypoint (tuple of (float, float)): the waypoint to track
        """
        # Find the waypoint to track
        # waypoint = (path[0].x, path[0].y)
        waypoint = None
        print("Path length: ", len(path))

        # Extract path a s tuples of (x, y)
        coordinates_path = np.array([[node.x, node.y] for node in path])
        
        # Pruning the path
        sub_paths = []
        for i in range(len(path) - 2):
            sub_path = coordinates_path
            for j in range(i + 2, len(path)):
                if not self.check_collision(path[i], path[j]):
                    sub_path = np.vstack((coordinates_path[:i+1], coordinates_path[j:]))
            sub_paths.append(sub_path)

        costs = np.array([np.linalg.norm(p[1:] - p[:-1], axis = 1).sum() for p in sub_paths])
        path = sub_paths[np.argmin(costs)]

        # Track the first waypoint of this path (after start)
        waypoint = path[1]
        return waypoint

    def compute_goal_point_in_car_frame(self, car_pose):
        """
        Compute the goal point on the waypoints.
        """
        car_x, car_y = car_pose.position.x, car_pose.position.y
        closest_index = np.argmin(np.linalg.norm(self.waypoints[:, :2] - np.array([car_x, car_y]), axis=1))
        goal_point = None
        for i in range(closest_index, closest_index + len(self.waypoints)):
            if i >= len(self.waypoints):
                i = i - len(self.waypoints)
            curr_dist = np.linalg.norm(self.waypoints[i, :2] - np.array([car_x, car_y]))
            if curr_dist > 0.8 * self.lookahead:
                goal_point = self.waypoints[i, :2]
                break
        
        goal_point_car_frame = self.transform_point_to_car_frame(goal_point, car_pose)
        # Try different shifts in y-axis to find a goal point that is not in an obstacle.
        shifts = np.array([0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.4, 0.4, -0.5, 0.5, -0.6, 0.6, -0.7, 0.7, -0.8, 0.8, -0.9, 0.9, -1.0, 1.0])
        for shift in shifts:
            modified_goal_point_car_frame = np.array([goal_point_car_frame[0], goal_point_car_frame[1] + shift])
            # Check if the straight line from the car to the goal point is in collision.
            if not self.occupancy_grid.check_line_collision(0, 0, modified_goal_point_car_frame[0], modified_goal_point_car_frame[1]):
                goal_point_car_frame = modified_goal_point_car_frame
                break

        return goal_point_car_frame


    def pose_callback(self, pose_msg):
        """
        The pose callback when subscrdibed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        """
        if not self.grid_formed:
            return

        # Get the current x, y position of the vehicle
        pose = pose_msg.pose.pose

        # Get the current goal
        goal_point_car_frame = self.compute_goal_point_in_car_frame(pose)
        is_obstacle = self.occupancy_grid.check_line_collision(0, 0, goal_point_car_frame[0], goal_point_car_frame[1])

        # Visualize the goal point
        if DEBUG:
            self.visualize_goal(goal_point_car_frame)

        new_rrt_required = False
        waypoint_to_track = goal_point_car_frame
        
        if is_obstacle:
            if self.current_rrt_delay_counter >= self.rrt_delay_counter or self.current_following_waypoint is None:
                new_rrt_required = True
                self.current_rrt_delay_counter = 0
            else:
                self.current_rrt_delay_counter += 1
                waypoint_to_track = self.transform_point_to_car_frame(self.current_following_waypoint, pose)
                new_rrt_required = False
        else:
            new_rrt_required = False
            self.current_rrt_delay_counter = 0
            self.current_following_waypoint = None

        # # Run RRT to get a new path if required.
        if new_rrt_required:            
            # path = self.rrt(goal_point_car_frame)
            path = self.rrt_star(goal_point_car_frame)
            if path is None or len(path) < 2:
                return None

            # Visualize the path
            if DEBUG:
                self.visualize_path(path)
                self.visualize_tree()

            # Find the waypoint to track
            waypoint_to_track = self.find_waypoint_to_track(path)
            if DEBUG:
                self.visualize_waypoint_to_track(waypoint_to_track)

            # Transform the waypoint to the map frame
            self.current_following_waypoint  = self.transform_point_to_map_frame(waypoint_to_track, pose)

        # Pure pursuit in car frame
        self.pure_pursuit(waypoint_to_track)


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
        
        # Sample using anormal distribution with mean as the center between the goal and the car
        # and standard deviation as the distance between the goal and the car / 2.
        mean = np.array([goal[0], goal[1]]) / 2
        std_dev = LA.norm(np.array([goal[0], goal[1]]) - np.array([0, 0])) / 2
        x = np.random.normal(mean[0], std_dev)
        y = np.random.normal(mean[1], std_dev/2)

        # Clip the sampled point to the grid bounds
        x = np.clip(x, 0.4, self.grid_bounds_x[1]*0.8)
        y = np.clip(y, self.grid_bounds_y[0]*0.8, self.grid_bounds_y[1]*0.8)

        # Check if the sampled point is in collision
        require_resample = False
        if self.occupancy_grid[x, y] == 1:
            require_resample = True

        if require_resample:
            return self.sample(goal)

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
        nearest_node = None
        nearest_dist = float('inf')
        for node in tree:
            dist = LA.norm(np.array([node.x, node.y]) - np.array(sampled_point))
            if dist < nearest_dist and dist > 1e-3:
                nearest_dist = dist
                nearest_node = node
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
        if dist_to_sampled > self.max_steer_distance:
            new_node = TreeNode()
            new_node.x = nearest_node.x + self.max_steer_distance * vec_to_sampled[0]
            new_node.y = nearest_node.y + self.max_steer_distance * vec_to_sampled[1]
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
        if (self.occupancy_grid.check_line_collision(nearest_node.x, nearest_node.y, new_node.x, new_node.y)):
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
    def calc_cost_new_node(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        # cost is the distance between the node and the root node
        # we check for all the nodes in the neighborhood of the node
        # and return the one with the least line cost to the current node
        if node.is_root:
            return 0
        neighborhood = self.near(tree, node)
        min_cost = float('inf')
        parent = None
        for n in neighborhood:
            cost = self.line_cost(n, node) + n.cost
            if cost < min_cost:
                # check if the line between n and node is in collision
                if self.check_collision(n, node):
                    continue
                min_cost = cost
                parent = n
            
        return min_cost, parent


    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (Node): node at one end of the straight line
            n2 (Node): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """
        # cost is the distance between n1 and n2 plus the cost of n1
        cost_between_nodes = LA.norm(np.array([n1.x, n1.y]) - np.array([n2.x, n2.y]))
        return cost_between_nodes


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
        for n in tree:
            dist = LA.norm(np.array([n.x, n.y]) - np.array([node.x, node.y]))
            if dist <= self.neighborhood_radius and dist > 1e-3:
                neighborhood.append(n)
        return neighborhood


    def rewire(self, tree, node):
        """
        This method should rewire the tree such that the cost of the nodes
        are minimized

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're rewiring the tree for
        """
        neighborhood = self.near(tree, node)
        for n in neighborhood:
            new_cost = self.line_cost(n, node) + node.cost
            if new_cost < n.cost:
                # check if the line between n and node is in collision
                if self.check_collision(n, node):
                    continue
                n.parent = node
                n.cost = new_cost


def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
