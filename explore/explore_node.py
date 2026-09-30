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
        self.min_gap_angle_degrees = float(self.declare_parameter('min_gap_angle_degrees', 60.0).value)

        # Publisher
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.twist = Twist()

        # State machine variables
        self.state = 'WAIT'
        self.action_start_time = None
        self.wait_timeout = 2.0
        self.last_reported_best_dir = None
        self.gap_stable_count = 0

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
            f"Spin speed: {self.spin_speed} rad/s, Min gap angle: {self.min_gap_angle_degrees}°"
        )

    def handle_reset(self, request, response):
        """Handle reset service calls - immediately transition to WAIT"""
        self.state = 'WAIT'
        self.action_start_time = None
        self.scan_message = None
        self.last_reported_best_dir = None
        self.gap_stable_count = 0

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
        MIN_GAP_ANGLE = math.radians(self.min_gap_angle_degrees)

        if self.scan_message is None or len(self.scan_message.ranges) == 0:
            return None, None

        ranges = self.scan_message.ranges
        range_min = self.scan_message.range_min
        range_max = self.scan_message.range_max
        angle_min = self.scan_message.angle_min
        angle_increment = self.scan_message.angle_increment

        # Merge consecutive .inf readings into single entries to properly detect large gaps
        valid_readings = []
        for i, distance in enumerate(ranges):
            angle = angle_min + i * angle_increment
            # .inf (infinity) means empty space beyond 12m - this is a VALID clear direction!
            is_valid = False
            if distance == float('inf'):
                # Infinity readings are valid - they mean no obstacle detected (empty space)
                is_valid = True
            elif distance >= range_min and distance < range_max * 0.95:
                # Valid finite distance reading within sensor range
                is_valid = True
            if is_valid:
                valid_readings.append((i, angle, distance))

        # Merge consecutive .inf readings to treat them as continuous voids
        merged_valid_readings = []
        if len(valid_readings) > 1:
            current_idx, current_angle, current_distance = valid_readings[0]
            
            for i in range(1, len(valid_readings)):
                next_idx, next_angle, next_dist = valid_readings[i]
                angular_diff = abs(next_angle - current_angle)
                
                if (next_idx != current_idx and 
                    current_distance == float('inf') and 
                    next_dist == float('inf')):
                    # Consecutive .inf readings with small angle gap - merge them
                    if angular_diff < math.radians(15):  # Less than 15° apart
                        continue  # Part of continuous .inf region
                else:
                    merged_valid_readings.append((current_idx, current_angle, current_distance))
                    current_idx, current_angle, current_distance = valid_readings[i]
            
            # Add the last entry
            merged_valid_readings.append((valid_readings[-1][0], valid_readings[-1][1], valid_readings[-1][2]))
            valid_readings = merged_valid_readings

        if len(valid_readings) < 2:
            return None, None

        best_gap_start_idx = None
        best_gap_end_idx = None
        max_gap_distance = 0.0

        for i in range(len(valid_readings) - 1):
            idx, angle, distance = valid_readings[i]
            next_idx, next_angle, next_dist = valid_readings[i + 1]
            gap_angle = (next_idx - idx) * angle_increment

            if gap_angle >= MIN_GAP_ANGLE:
                # Handle .inf values: treat as maximum clear path for stability
                if distance == float('inf') and next_dist == float('inf'):
                    avg_gap_distance = range_max  # Both edges empty - use max range
                elif distance == float('inf') or next_dist == float('inf'):
                    # One edge is infinity, use the finite one capped at range_max
                    finite_edge = distance if distance != float('inf') else next_dist
                    avg_gap_distance = min(finite_edge, range_max)
                else:
                    avg_gap_distance = (distance + next_dist) / 2.0
                if avg_gap_distance > max_gap_distance:
                    max_gap_distance = avg_gap_distance
                    best_gap_start_idx = idx
                    best_gap_end_idx = next_idx

        if best_gap_end_idx is not None:
            gap_center_idx = (best_gap_start_idx + best_gap_end_idx) / 2.0
            gap_center_absolute = angle_min + gap_center_idx * angle_increment
            rotation_needed = gap_center_absolute - angle_min
        else:
            return None, None

        return rotation_needed, max_gap_distance

    def find_closest_obstacle(self):
        """
        Find closest obstacle in any direction (no cone filtering).
        
        This finds the absolute nearest valid LIDAR reading to trigger emergency stop.
        If ANY obstacle is within stop_distance, we abort immediately for safety.
        """
        if self.scan_message is None or len(self.scan_message.ranges) == 0:
            return None

        ranges = self.scan_message.ranges
        range_min = self.scan_message.range_min
        range_max = self.scan_message.range_max
        
        # Find closest obstacle (any direction, no cone filtering for safety)
        min_distance = float('inf')
        
        for i, distance in enumerate(ranges):
            is_valid = (distance < range_max * 0.95) and (distance >= range_min)
            
            if is_valid and distance < min_distance:
                min_distance = distance

        result = min_distance if min_distance < float('inf') else None
        
        if result is not None:
            self.get_logger().info(f"DEBUG: find_closest_obstacle - closest obstacle at {result:.2f}m")
        return result

    def drive_callback(self):
        """Main timer callback - state machine logic"""
        now = self.clock.now().nanoseconds / 1e9
        self.get_logger().info(f"DEBUG [timer]: State={self.state}")

        if self.state == 'WAIT':
            return

        elif self.state == 'EXPLORE':
            obstacle_distance = self.find_closest_obstacle()

            if obstacle_distance is not None and obstacle_distance <= self.stop_distance:
                self.get_logger().info(f"OBSTACLE DETECTED at {obstacle_distance:.2f}m, transitioning to OBSTACLE_AVOID")
                self.state = 'OBSTACLE_AVOID'
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                self.get_logger().info(f"DEBUG [PUBLISH]: State changed to OBSTACLE_AVOID, twist={self.twist.linear.x:.2f}, {self.twist.angular.z:.2f}"
                )

                self.publisher_.publish(self.twist)
            else:
                self.get_logger().info("EXPLORE: Clear path ahead, moving forward")
                self.twist.linear.x = self.linear_speed
                self.twist.angular.z = 0.0
                self.get_logger().info(f"DEBUG [PUBLISH]: Moving forward, twist={self.twist.linear.x:.2f}, {self.twist.angular.z:.2f}"
                )
                
                self.publisher_.publish(self.twist)

                duration = self.move_distance / self.linear_speed
                if self.action_start_time is not None and (now - self.action_start_time >= duration):
                    self.get_logger().info(f"EXPLORE movement complete after {duration:.2f}s, continuing forward")
                    self.action_start_time = now

        elif self.state == 'OBSTACLE_AVOID':
            best_direction, gap_distance = self.find_best_escape_path()

            # Handle case where no valid escape path is found (None, None)
            if best_direction is not None and gap_distance is not None:
                self.get_logger().info(
                    f"DEBUG OBSTACLE_AVOID: best_direction={best_direction:.2f} rad ({math.degrees(best_direction):.1f}°), "
                    f"gap_distance={gap_distance:.2f}m, stop_distance*2={self.stop_distance*2:.2f}m"
                )

            if best_direction is not None and gap_distance > self.stop_distance * 2:
                self.gap_stable_count = getattr(self, 'gap_stable_count', 0) + 1
                
                MIN_STABLE_SCANS = 3
                self.get_logger().info(
                    f"DEBUG OBSTACLE_AVOID: Wide gap ({math.degrees(best_direction):.1f}°, {gap_distance:.2f}m) "
                    f"{self.gap_stable_count}/{MIN_STABLE_SCANS} consecutive scans confirmed."
                )

                if self.gap_stable_count >= MIN_STABLE_SCANS:
                    current_angular_vel = self.angular_speed * math.copysign(1, best_direction)
                    self.get_logger().info(
                        f"DEBUG OBSTACLE_AVOID: Path confirmed stable! Rotating toward {math.degrees(best_direction):.1f}°, "
                        f"gap distance={gap_distance:.2f}m (threshold > {self.stop_distance*2:.1f}m)"
                    )
                    self.twist.linear.x = self.linear_speed
                    self.twist.angular.z = 0.0
                    self.publisher_.publish(self.twist)  # FIXED: Publish twist to rotate toward gap
                    self.get_logger().info(f"DEBUG [PUBLISH]: Transitioning back to EXPLORE, rotating toward {math.degrees(best_direction):.1f}°, "f"twist={self.twist.linear.x:.2f}, {self.twist.angular.z:.2f}")

                    self.state = 'EXPLORE'
                    self.action_start_time = now
                    self.gap_stable_count = 0
                else:
                    current_angular_vel = -self.spin_speed * 0.3
                    self.get_logger().info(
                        f"DEBUG OBSTACLE_AVOID: Gap not yet stable ({self.gap_stable_count} scans), "
                        f"continuing scan at {abs(current_angular_vel):.2f} rad/s"
                    )
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = current_angular_vel  # Only spin, no forward motion
                    self.get_logger().info(f"DEBUG [PUBLISH]: Spinning to scan, twist={self.twist.linear.x:.2f}, {self.twist.angular.z:.2f}")

                    self.publisher_.publish(self.twist)  # FIXED: Publish twist to spin

                if best_direction is not None and best_direction != self.last_reported_best_dir:
                    self.get_logger().info(f"DEBUG OBSTACLE_AVOID: [Best] {math.degrees(best_direction):.1f}°, dist={gap_distance:.2f}m")
                    self.last_reported_best_dir = best_direction

            else:
                # Case 1: No valid escape path found (None, None)
                # Case 2: Escape path exists but gap is too narrow
                if best_direction is None:
                    current_angular_vel = -self.spin_speed * 0.5  # Spin faster when no path at all
                    self.get_logger().warning(
                        f"DEBUG OBSTACLE_AVOID: NO VALID ESCAPE PATH FOUND! "
                        f"Spinning rapidly at {abs(current_angular_vel):.2f} rad/s to scan for opening"
                    )
                else:
                    current_angular_vel = -self.spin_speed * 0.3
                    self.get_logger().warning(
                        f"DEBUG OBSTACLE_AVOID: Gap too narrow ({gap_distance:.2f}m < {self.stop_distance*2:.1f}m), "
                        f"spinning at {abs(current_angular_vel):.2f} rad/s to find better direction"
                    )
                self.twist.linear.x = 0.0
                self.twist.angular.z = current_angular_vel
                self.get_logger().info(f"DEBUG [PUBLISH]: No valid path, spinning rapidly, twist={self.twist.linear.x:.2f}, {self.twist.angular.z:.2f}")
                self.publisher_.publish(self.twist)  # FIXED: Publish twist to spin

            if best_direction is None or (gap_distance <= self.stop_distance * 2 if best_direction else False):
                self.last_reported_best_dir = None

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
