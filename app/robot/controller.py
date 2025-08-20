import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Comandos válidos y sus parámetros de movimiento
COMMAND_MAP = {
    "forward":     {"linear": 0.5, "angular": 0.0},
    "back":        {"linear": -0.5, "angular": 0.0},
    "turn_left":   {"linear": 0.0, "angular": 0.5},
    "turn_right":  {"linear": 0.0, "angular": -0.5},
    "stand":       {"linear": 0.0, "angular": 0.0},
    "sit":         {"linear": 0.0, "angular": 0.0},
}

class RobotController:
    def __init__(self):
        self._ros_initialized = False
        self._node = None
        self._publisher = None

    def init_ros(self):
        if not self._ros_initialized:
            rclpy.init()
            self._node = rclpy.create_node('hexamind_controller')
            self._publisher = self._node.create_publisher(Twist, '/cmd_vel', 10)
            self._ros_initialized = True

    def perform_move(self, cmd: str) -> dict:
        if cmd not in COMMAND_MAP:
            return {"ok": False, "error": "Comando inválido", "executed": cmd}

        self.init_ros()

        twist = Twist()
        twist.linear.x = COMMAND_MAP[cmd]["linear"]
        twist.angular.z = COMMAND_MAP[cmd]["angular"]

        self._publisher.publish(twist)
        self._node.get_logger().info(f"Moviendo: {cmd}")

        return {"ok": True, "executed": cmd}

# Instancia compartida
robot = RobotController()

def perform_move(cmd: str) -> dict:
    return robot.perform_move(cmd)