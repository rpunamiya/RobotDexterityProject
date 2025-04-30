#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
import cv2
import numpy as np
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import tf2_ros
from rclpy.duration import Duration

# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper
from geometry_msgs.msg import WrenchStamped
import math
import pandas as pd
import matplotlib.pyplot as plt


class ForceSensing(Node):
    def __init__(self):
        super().__init__('Force_Sensing_Node')

        # Initialize variables to store the latest force/torque data
        self.FT_force_x = 0.0
        self.FT_force_y = 0.0
        self.FT_force_z = 0.0
        self.FT_torque_x = 0.0
        self.FT_torque_y = 0.0
        self.FT_torque_z = 0.0
        
        # Create a subscription to the force/torque sensor topic
        self.ft_ext_state_sub = self.create_subscription(WrenchStamped, '/xarm/uf_ftsensor_ext_states', self.ft_ext_state_cb, 10)
        
        # Create a timer that calls move_log_ft every 1.0 seconds (1 Hz)
        self.timer = self.create_timer(5.0, self.move_log_ft)
        
        # Log a message to indicate the node has started
        self.get_logger().info("FT Monitor started - logging at 1.0 Hz")
        
        # Replace the direct publishers with the command queue publisher
        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)
        
        # Subscribe to current arm pose and gripper position for status tracking (optional)
        self.current_arm_pose = None
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.arm_pose_callback, 10)
        
        self.current_gripper_position = None
        self.gripper_status_sub = self.create_subscription(Float64, '/me314_xarm_gripper_position', self.gripper_position_callback, 10)

        self.init_arm_pose = None
        self.contact_pose = None
        self.final_pose = None

        self.strain = []
        self.stress = []

    def arm_pose_callback(self, msg: Pose):
        self.current_arm_pose = msg

    def gripper_position_callback(self, msg: Float64):
        self.current_gripper_position = msg.data

    def publish_pose(self, pose_array: list):
        """
        Publishes a pose command to the command queue using an array format.
        pose_array format: [x, y, z, qx, qy, qz, qw]
        """
        # Create a CommandQueue message containing a single pose command
        queue_msg = CommandQueue()
        queue_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Create a CommandWrapper for the pose command
        wrapper = CommandWrapper()
        wrapper.command_type = "pose"
        
        # Populate the pose_command with the values from the pose_array
        wrapper.pose_command.x = pose_array[0]
        wrapper.pose_command.y = pose_array[1]
        wrapper.pose_command.z = pose_array[2]
        wrapper.pose_command.qx = pose_array[3]
        wrapper.pose_command.qy = pose_array[4]
        wrapper.pose_command.qz = pose_array[5]
        wrapper.pose_command.qw = pose_array[6]
        
        # Add the command to the queue and publish
        queue_msg.commands.append(wrapper)
        self.command_queue_pub.publish(queue_msg)
        
        self.get_logger().info(f"Published Pose to command queue:\n"
                               f"  position=({pose_array[0]}, {pose_array[1]}, {pose_array[2]})\n"
                               f"  orientation=({pose_array[3]}, {pose_array[4]}, "
                               f"{pose_array[5]}, {pose_array[6]})")

    def publish_gripper_position(self, gripper_pos: float):
        """
        Publishes a gripper command to the command queue.
        For example:
          0.0 is "fully open"
          1.0 is "closed"
        """
        # Create a CommandQueue message containing a single gripper command
        queue_msg = CommandQueue()
        queue_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Create a CommandWrapper for the gripper command
        wrapper = CommandWrapper()
        wrapper.command_type = "gripper"
        wrapper.gripper_command.gripper_position = gripper_pos
        
        # Add the command to the queue and publish
        queue_msg.commands.append(wrapper)
        self.command_queue_pub.publish(queue_msg)
        
        self.get_logger().info(f"Published gripper command to queue: {gripper_pos:.2f}")
    
    def ft_ext_state_cb(self, msg: WrenchStamped):
        """
        Callback function that runs whenever a new force/torque message is received.
        
        This function extracts the force and torque data from the message
        and stores it for later use.
        
        Args:
            msg (WrenchStamped): The force/torque sensor message
        """
        if self.init_arm_pose is None:
            self.init_arm_pose = self.current_arm_pose
        
        # Extract force components from the message
        self.FT_force_x = msg.wrench.force.x
        self.FT_force_y = msg.wrench.force.y
        self.FT_force_z = msg.wrench.force.z
        
        # Extract torque components from the message
        self.FT_torque_x = msg.wrench.torque.x
        self.FT_torque_y = msg.wrench.torque.y
        self.FT_torque_z = msg.wrench.torque.z

    def move_log_ft(self):
        """
        Timer callback function that logs force/torque data at a fixed rate (1 Hz).
        
        This function:
        1. Calculates the magnitude of force and torque vectors
        2. Logs the individual components and magnitudes
        """
        # Calculate force magnitude using the Euclidean norm (square root of sum of squares)
        force_magnitude = math.sqrt(self.FT_force_x**2 + self.FT_force_y**2 + self.FT_force_z**2)
        
        # Calculate torque magnitude
        # torque_magnitude = math.sqrt(self.FT_torque_x**2 + self.FT_torque_y**2 + self.FT_torque_z**2)
        
        # Log the force data (components and magnitude)
        self.get_logger().info(f"Force: [{self.FT_force_x:.2f}, {self.FT_force_y:.2f}, {self.FT_force_z:.2f}] N")
        self.get_logger().info(f"Force magnitude: {force_magnitude:.2f} N")

        if self.contact_pose is not None:
            displacement = self.current_arm_pose.position.z - self.contact_pose.position.z
            self.get_logger().info(f"Displacement: {displacement:.2f} m")

            width = 5.1 / 1000
            length = 32.2 / 1000
            area = width * length
            stress = force_magnitude / area
            strain = displacement / 0.0508  # original ball length of 0.0508m
            self.strain.append(strain)
            self.stress.append(stress)
        
        
        # Log the torque data (components and magnitude)
        # self.get_logger().info(f"Torque: [{self.FT_torque_x:.2f}, {self.FT_torque_y:.2f}, {self.FT_torque_z:.2f}] Nm")
        # self.get_logger().info(f"Torque magnitude: {torque_magnitude:.2f} Nm")

        if self.contact_pose is None:
            if force_magnitude > 0.01:
                self.contact_pose = self.current_arm_pose
                self.get_logger().info("Contact detected.")
            else:
                self.get_logger().info("No contact detected.")
                # move the robot down
                self.get_logger().info("Current pose: "
                                       f"[{self.current_arm_pose.position.x:.2f}, "
                                       f"{self.current_arm_pose.position.y:.2f}, "
                                       f"{self.current_arm_pose.position.z:.2f}]")
                self.publish_pose([self.current_arm_pose.position.x,
                                   self.current_arm_pose.position.y,
                                   self.current_arm_pose.position.z - 0.05,
                                   self.current_arm_pose.orientation.x,
                                   self.current_arm_pose.orientation.y,
                                   self.current_arm_pose.orientation.z,
                                   self.current_arm_pose.orientation.w])
        else:
            if force_magnitude < 2:
                self.get_logger().info("Force is below 2N, moving robot 0.01m down.")
                self.publish_pose([self.current_arm_pose.position.x,
                                   self.current_arm_pose.position.y,
                                   self.current_arm_pose.position.z - 0.01,
                                   self.current_arm_pose.orientation.x,
                                   self.current_arm_pose.orientation.y,
                                   self.current_arm_pose.orientation.z,
                                   self.current_arm_pose.orientation.w])
            elif force_magnitude < 5:
                self.get_logger().info("Force is between 2N and 5N, moving robot 0.001m down.")
                self.publish_pose([self.current_arm_pose.position.x,
                                   self.current_arm_pose.position.y,
                                   self.current_arm_pose.position.z - 0.001,
                                   self.current_arm_pose.orientation.x,
                                   self.current_arm_pose.orientation.y,
                                   self.current_arm_pose.orientation.z,
                                   self.current_arm_pose.orientation.w])
            else:
                self.get_logger().info("Force is above 5N, stopped the robot.")
                if self.final_pose is None:
                    self.final_pose = self.current_arm_pose
                    self.get_logger().info("Final pose set.")

        # Wait until key press
        cv2.waitKey(1)
        

def main(args=None):
    rclpy.init(args=args)
    node = ForceSensing()
    
    # Let's first close the gripper (0.0 to 1.0, where 0.0 is fully open and 1.0 is fully closed)
    node.get_logger().info("Closing gripper...")
    node.publish_gripper_position(1.0)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down node.")
    finally:

        node.destroy_node()
        rclpy.shutdown()

    node.get_logger().info("All actions done. Shutting down.")
    node.publish_pose([node.init_arm_pose.position.x, node.init_arm_pose.position.y,
                       node.init_arm_pose.position.z, node.init_arm_pose.orientation.x,
                       node.init_arm_pose.orientation.y, node.init_arm_pose.orientation.z,
                       node.init_arm_pose.orientation.w])
    node.publish_gripper_position(0.0)  # Open the gripper
    node.get_logger().info("Gripper opened and robot moved to initial pose.")

    # Save the strain and stress data to a csv file
    data = {
        'strain': node.strain,
        'stress': node.stress
    }
    df = pd.DataFrame(data)
    df.to_csv('strain_stress_data.csv', index=False)
    node.get_logger().info("Strain and stress data saved to strain_stress_data.csv.")

    node.destroy_node()
    rclpy.shutdown()
    
    # Read the csv file and plot the stress vs strain curve
    df = pd.read_csv('strain_stress_data.csv')
    plt.plot(df['strain'], df['stress'])
    plt.xlabel('Strain')
    plt.ylabel('Stress (Pa)')
    plt.title('Stress vs Strain Curve')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()