#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger
from sensor_msgs.msg import LaserScan
import math


class ExploreNode(Node):
    """
    Explore node with 3 states: WAIT, EXPLORE, OBSTACLE_AVOID
    
    - WAIT: Listen for start trigger, wait indefinitely
    - EXPLORE: Move forward like simple_drive FORWARD_1
    - OBSTACLE_AVOID: Analyze 1080-point LIDAR, find clear path, spin to it

    When obstacle detected → transition to OBSTACLE_AVOID
    If no path found → go back to WAIT
    Reset trigger → immediate transition to WAIT
    """

    def __init__(self):
        super().__init__('explore')

        # Parameters
        self.linear_speed = float(self.declare_parameter('linear_speed', 0.2).value)
        self.angular_speed = float(self.declare_parameter('angular_speed', 0.5).value)
        self.move_distance = float(self.declare_parameter('move_distance', 1.25).value)
        self.stop_distance = float(self.declare_parameter('stop_distance', 0.5).value)
        self.spin_speed = float(self.declare_parameter('spin_speed', 1.0).value)
        self.min_gap_angle = float(self.declare_parameter('min_gap_angle_degrees', 60.0).value)

        # Publisher
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.twist = Twist()

        # State machine variables
        self.state = 'WAIT'
        self.action_start_time = None
        self.wait_timeout = 2.0
        self.last_reported_best_dir = None

        # LIDAR subscription (30Hz polling for responsiveness)
        self.scan_message = None
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Clock and timer (500ms polling - only need to check when moving)
        self.clock = self.get_clock()
        self.timer_period = 0.5
        self.timer = self.create_timer(self.timer_period, self.drive_callback)

        # Services
        self.reset_service = self.create_service(Trigger, '/explore/reset', self.handle_reset)
        self.start_service = self.create_service(Trigger, '/explore/start', self.handle_start)

        self.get_logger().info(
            f"Explore Node initialized. "
            f"Linear speed: {self.linear_speed} m/s, Angular speed: {self.angular_speed} rad/s, "
            f"Move distance: {self.move_distance}m, Stop distance: {self.stop_distance}m, "
            f"Spin speed: {self.spin_speed} rad/s, Min gap angle: {self.min_gap_angle}°"
        )

    def handle_reset(self, request, response):
        """Handle reset service calls - immediately transition to WAIT"""
        self.state = 'WAIT'
        self.action_start_time = None
        self.scan_message = None
        self.last_reported_best_dir = None

        # Clear any pending velocity commands
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.publisher_.publish(self.twist)

        self.get_logger().info("RESET service called, robot returned to WAIT state")
        response.success = True
        response.message = "Robot reset successfully - now in WAIT state"
        return response

    def handle_start(self, request, response):
        """Handle start service calls - transition to EXPLORE"""
        self.state = 'EXPLORE'
        self.action_start_time = self.clock.now().nanoseconds / 1e9

        # Clear any pending velocity commands
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.publisher_.publish(self.twist)

        self.get_logger().info("START service called, transitioning to EXPLORE state")
        response.success = True
        response.message = "Starting exploration"
        return response

    def scan_callback(self, msg):
        """Callback when new LIDAR scan data is received."""
        if len(msg.ranges) > 0:
            self.get_logger().info(f"New LIDAR scan received with {len(msg.ranges)} rays")
        else:
            self.get_logger().warning("Received empty LIDAR scan")
        self.scan_message = msg

    def find_best_escape_path(self):
        """
        Analyze full 360° LIDAR scan (1080 points) to find the direction with longest clear path.
        
        Returns:
            best_rotation_angle: Angle to rotate toward gap center (-π to +π), or None
            max_gap_distance: Distance in that direction, or None
        """
        MIN_GAP_ANGLE = math.radians(self.min_gap_angle)  # Convert degrees to radians

        if self.scan_message is None or len(self.scan_message.ranges) == 0:
            return None, None

        ranges = self.scan_message.ranges
        range_min = self.scan_message.range_min
        range_max = self.scan_message.range_max
        angle_min = self.scan_message.angle_min
        angle_increment = self.scan_message.angle_increment

        # Filter valid readings (not noise, not beyond max range)
        valid_readings = []
        for i, distance in enumerate(ranges):
            angle = angle_min + i * angle_increment
            is_valid = (distance < range_max * 0.95) and (distance >= range_min)
            if is_valid:
                valid_readings.append((i, angle, distance))

        if len(valid_readings) < 2:
            return None, None

        # Find the largest continuous gap meeting MIN_GAP_ANGLE width
        best_gap_start_idx = None
        best_gap_end_idx = None
        max_gap_distance = 0.0

        for i in range(len(valid_readings) - 1):
            idx, angle, distance = valid_readings[i]
            next_idx, next_angle, next_dist = valid_readings[i + 1]

            # Calculate gap width (angle span between rays)
            gap_angle = (next_idx - idx) * angle_increment

            # Check if gap meets minimum width requirement
            if gap_angle >= MIN_GAP_ANGLE:
                avg_gap_distance = (distance + next_dist) / 2.0

                if avg_gap_distance > max_gap_distance:
                    max_gap_distance = avg_gap_distance
                    best_gap_start_idx = idx
                    best_gap_end_idx = next_idx

        # Calculate gap center angle in LIDAR frame
        if best_gap_end_idx is not None:
            gap_center_idx = (best_gap_start_idx + best_gap_end_idx) / 2.0
            gap_center_absolute = angle_min + gap_center_idx * angle_increment
            rotation_needed = gap_center_absolute - angle_min
        else:
            return None, None

        return rotation_needed, max_gap_distance

    def drive_callback(self):
        """Main timer callback - state machine logic"""
        now = self.clock.now().nanoseconds / 1e9
        self.get_logger().info(f"DEBUG [timer]: State={self.state}")

        # WAIT state: do nothing, just wait for start trigger
        if self.state == 'WAIT':
            return

        # EXPLORE state: move forward like simple_drive FORWARD_1
        elif self.state == 'EXPLORE':
            # Check for obstacle - emergency stop if detected
            obstacle_distance = self.find_closest_obstacle()

            if obstacle_distance is not None and obstacle_distance <= self.stop_distance:
                self.get_logger().info(f"OBSTACLE DETECTED at {obstacle_distance:.2f}m, transitioning to OBSTACLE_AVOID")
                self.state = 'OBSTACLE_AVOID'
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                self.publisher_.publish(self.twist)
            else:
                self.get_logger().info("EXPLORE: Clear path ahead, moving forward")
                self.twist.linear.x = self.linear_speed
                self.twist.angular.z = 0.0
                self.publisher_.publish(self.twist)

                duration = self.move_distance / self.linear_speed
                if self.action_start_time is not None and (now - self.action_start_time >= duration):
                    self.get_logger().info(f"EXPLORE movement complete after {duration:.2f}s, continuing forward")
                    # Reset timer for next move distance
                    self.action_start_time = now

        # OBSTACLE_AVOID state: analyze LIDAR and find clear path
        elif self.state == 'OBSTACLE_AVOID':
            best_direction, gap_distance = self.find_best_escape_path()

            # Log the analysis results
            self.get_logger().info(
                f"DEBUG OBSTACLE_AVOID: best_direction={best_direction:.2f} rad ({math.degrees(best_direction):.1f}°), "
                f"gap_distance={gap_distance:.2f}m, stop_distance*2={self.stop_distance*2:.2f}m"
            )

            if best_direction is not None and gap_distance > self.stop_distance * 2:
                # Wide gap found - update stability tracking
                self.gap_stable_count = getattr(self, 'gap_stable_count', 0) + 1
                
                MIN_STABLE_SCANS = 3
                self.get_logger().info(
                    f"DEBUG OBSTACLE_AVOID: Wide gap ({math.degrees(best_direction):.1f}°, {gap_distance:.2f}m) "
                    f"{self.gap_stable_count}/{MIN_STABLE_SCANS} consecutive scans confirmed."
                )

                if self.gap_stable_count >= MIN_STABLE_SCANS:
                    # Commit to path - rotate minimally toward it and move forward
                    current_angular_vel = self.angular_speed * math.copysign(1, best_direction)
                    self.get_logger().info(
                        f"DEBUG OBSTACLE_AVOID: Path confirmed stable! Rotating toward {math.degrees(best_direction):.1f}°, "
                        f"gap distance={gap_distance:.2f}m (threshold > {self.stop_distance*2:.1f}m)"
                    )
                    self.twist.linear.x = self.linear_speed  # Commit to forward motion
                    self.twist.angular.z = current_angular_vel
                    self.state = 'EXPLORE'  # Transition back to EXPLORE
                    self.action_start_time = now
                    self.gap_stable_count = 0  # Reset for next avoidance event
                else:
                    # Keep scanning - gentle spin to acquire more data
                    current_angular_vel = -self.spin_speed * 0.3
                    self.get_logger().info(
                        f"DEBUG OBSTACLE_AVOID: Gap not yet stable ({self.gap_stable_count} scans), "
                        f"continuing scan at {abs(current_angular_vel):.2f} rad/s"
                    )
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = current_angular_vel

                # Update last reported best direction to prevent repeated logging
                if best_direction != self.last_reported_best_dir:
                    self.get_logger().info(f"DEBUG OBSTACLE_AVOID: [Best] {math.degrees(best_direction):.1f}°, dist={gap_distance:.2f}m")
                    self.last_reported_best_dir = best_direction

            else:
                # No wide gap found or below safety threshold - gentle spinning scan
                current_angular_vel = -self.spin_speed * 0.3
                self.get_logger().warning(
                    f"DEBUG OBSTACLE_AVOID: No clear path (> {self.stop_distance*2:.1f}m), "
                    f"spinning at {abs(current_angular_vel):.2f} rad/s to find gap"
                )
                self.twist.linear.x = 0.0
                self.get_logger().info(
                    f"DEBUG OBSTACLE_AVOID: No clear path (> {self.stop_distance*2:.1f}m), "
                    f"spinning at {abs(current_angular_vel):.2f} rad/s to find gap"
                )

            # Reset last reported best dir when not committed to a path
            if best_direction is None or (gap_distance <= self.stop_distance * 2 if best_direction else False):
                self.last_reported_best_dir = None

    def find_closest_obstacle(self):
        """Find closest obstacle in forward direction."""
        if self.scan_message is None or len(self.scan_message.ranges) == 0:
            return None

        ranges = self.scan_message.ranges
        max_range = self.scan_message.range_max

        # Collect all valid readings (forward direction only for simplicity)
        valid_readings = []
        for i, distance in enumerate(ranges):
            angle = getattr(self.scan_message, 'angle_min', -math.pi) + i * getattr(self.scan_message, 'angle_increment', math.pi/180.0)
            is_valid = (distance < max_range * 0.95) and (distance >= self.scan_message.range_min)
            if is_valid:
                valid_readings.append((i, angle, distance))

        # Find closest obstacle
        min_distance = float('inf')
        for i, angle, distance in valid_readings:
            if distance < min_distance:
                min_distance = distance

        result = min_distance if min_distance < float('inf') else None
        return result

    def stop_and_wait(self):
        """Stop the robot and transition to WAIT state"""
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.publisher_.publish(self.twist)
        self.state = 'WAIT'
        self.get_logger().info("Transitioning to WAIT state")


def main(args=None):
    rclpy.init()
    node = ExploreNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()