from abc import abstractmethod
import time

import numpy as np
from pyquaternion import Quaternion

from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.const import ConfigurationPathAlgorithms as ObjectType
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.const import ObjectType
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.robots.arms.arm import Arm
from pyrep.robots.end_effectors.gripper import Gripper

from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.robot import Robot
from rlbench.backend.robot import UnimanualRobot
from rlbench.backend.robot import BimanualRobot
from rlbench.backend.scene import Scene
from rlbench.const import SUPPORTED_ROBOTS

import logging

from abc import ABC



def assert_action_shape(action: np.ndarray, expected_shape: tuple):
    if np.shape(action) != expected_shape:
        raise InvalidActionError(
            'Expected the action shape to be: %s, but was shape: %s' % (
                str(expected_shape), str(np.shape(action))))


def assert_unit_quaternion(quat):
    if not np.isclose(np.linalg.norm(quat), 1.0):
        raise InvalidActionError('Action contained non unit quaternion!')


def calculate_delta_pose(robot: Robot, action: np.ndarray):
    a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
    x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
    new_rot = Quaternion(
        a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
    qw, qx, qy, qz = list(new_rot)
    pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
    return pose


class ArmActionMode(ABC):
    
    _callable_each_step = None

    def action(self, scene: Scene, action: np.ndarray):
        self.action_pre_step(scene, action)
        self.action_step(scene)
        self.action_post_step(scene, action)    

    def action_step(self, scene: Scene):
        scene.step()
        if self._callable_each_step is not None:
            self._callable_each_step(scene.get_observation())

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        pass

    def action_post_step(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass

    def set_control_mode(self, robot: Robot):
        if isinstance(robot, UnimanualRobot):
            robot.arm.set_control_loop_enabled(True)
        elif isinstance(robot, BimanualRobot):
            logging.info("Setting control mode for both robots")
            robot.right_arm.set_control_loop_enabled(True)
            robot.left_arm.set_control_loop_enabled(True)

    def record_end(self, scene, steps=60, step_scene=True):
        if self._callable_each_step is not None:
            for _ in range(steps):
                if step_scene:
                    scene.step()
                self._callable_each_step(scene.get_observation())

    def set_callable_each_step(self, callable_each_step):
        self._callable_each_step = callable_each_step

class JointVelocity(ArmActionMode):
    """Control the joint velocities of the arm.

    Similar to the action space in many continious control OpenAI Gym envs.
    """

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        scene.robot.arm.set_joint_target_velocities(action)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],

    def set_control_mode(self, robot: Robot):
        robot.arm.set_control_loop_enabled(False)
        robot.arm.set_motor_locked_at_zero_velocity(True)


class BimanualJointVelocity(ArmActionMode): 

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        right_action = action[:7]
        left_action = action[7:]
        scene.robot.right_arm.set_joint_target_velocities(right_action)
        scene.robot.left_arm.set_joint_target_velocities(left_action)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))
        right_action = action[:7]
        left_action = action[7:]
        scene.robot.right_arm.set_joint_target_velocities(np.zeros_like(right_action))
        scene.robot.left_arm.set_joint_target_velocities(np.zeros_like(left_action))

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],

    def set_control_mode(self, robot: Robot):
        robot.right_arm.set_control_loop_enabled(False)
        robot.right_arm.set_motor_locked_at_zero_velocity(True)
        robot.left_arm.set_control_loop_enabled(False)
        robot.left_arm.set_motor_locked_at_zero_velocity(True)
        

class BimanualJointPosition(ArmActionMode):

    def __init__(self, absolute_mode: bool = True):
        self._absolute_mode = absolute_mode

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        
        right_action = action[:7]
        left_action = action[7:]

        if not self._absolute_mode:
            right_action = np.array(scene.robot.right_arm.get_joint_positions()) + right_action
            left_action = np.array(scene.robot.left_arm.get_joint_positions()) + left_action
            
        scene.robot.right_arm.set_joint_target_positions(right_action)
        scene.robot.left_arm.set_joint_target_positions(left_action)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.right_arm.set_joint_target_positions(
            scene.robot.right_arm.get_joint_positions())
        scene.robot.left_arm.set_joint_target_positions(
            scene.robot.left_arm.get_joint_positions())
    
    def action_shape(self, scene: Scene) -> tuple:
        return (14, )
        #return SUPPORTED_ROBOTS[scene.robot_setup][2],

    def unimanual_action_shape(self, scene: Scene) -> tuple:
        return (7, )



class JointPosition(ArmActionMode):
    """Control the target joint positions (absolute or delta) of the arm.

    The action mode opoerates in absolute mode or delta mode, where delta
    mode takes the current joint positions and adds the new joint positions
    to get a set of target joint positions. The robot uses a simple control
    loop to execute until the desired poses have been reached.
    It os the users responsibility to ensure that the action lies within
    a usuable range.
    """

    def __init__(self, absolute_mode: bool = True):
        """
        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
        """
        self._absolute_mode = absolute_mode


    def action_pre_step(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        if not  self._absolute_mode :
            action = np.array(scene.robot.arm.get_joint_positions()) + action
        scene.robot.arm.set_joint_target_positions(action)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.arm.set_joint_target_positions(
            scene.robot.arm.get_joint_positions())

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],



class JointTorque(ArmActionMode):
    """Control the joint torques of the arm.
    """

    TORQUE_MAX_VEL = 9999

    def _torque_action(self, robot, action):
        tml = JointTorque.TORQUE_MAX_VEL
        robot.arm.set_joint_target_velocities(
            [(tml if t < 0 else -tml) for t in action])
        robot.arm.set_joint_forces(np.abs(action))

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        self._torque_action(scene.robot, action)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        self._torque_action(scene.robot, scene.robot.arm.get_joint_forces())
        scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],


class EndEffectorPoseViaPlanning(ArmActionMode):
    """High-level action where target pose is given and reached via planning.

    Given a target pose, a linear path is first planned (via IK). If that fails,
    sample-based planning will be used. The decision to apply collision
    checking is a crucial trade off! With collision checking enabled, you
    are guaranteed collision free paths, but this may not be applicable for task
    that do require some collision. E.g. using this mode on pushing object will
    mean that the generated path will actively avoid not pushing the object.

    Note that path planning can be slow, often taking a few seconds in the worst
    case.

    This was the action mode used in:
    James, Stephen, and Andrew J. Davison. "Q-attention: Enabling Efficient
    Learning for Vision-based Robotic Manipulation."
    arXiv preprint arXiv:2105.14829 (2021).
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: str = 'world',
                 collision_checking: bool = False):
        """
        If collision check is enbled, and an object is grasped, then we

        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either 'world' or 'end effector'.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        self._callable_each_step = None
        self._robot_shapes = None

        if frame not in ['world', 'end effector']:
            raise ValueError("Expected frame to one of: 'world, 'end effector'")

    def _quick_boundary_check(self, scene: Scene, action: np.ndarray):
        pos_to_check = action[:3]
        relative_to = None if self._frame == 'world' else scene.robot.arm.get_tip()
        if relative_to is not None:
            scene.target_workspace_check.set_position(pos_to_check, relative_to)
            pos_to_check = scene.target_workspace_check.get_position()
        if not scene.check_target_in_workspace(pos_to_check):
            raise InvalidActionError('A path could not be found because the '
                                     'target is outside of workspace.')

    def _pose_in_end_effector_frame(self, robot: Robot, action: np.ndarray):
        a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
        x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
        new_rot = Quaternion(
            a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
        qw, qx, qy, qz = list(new_rot)
        pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
        return pose
    
    def set_callable_each_step(self, callable_each_step):
        self._callable_each_step = callable_each_step


    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        path = self.get_path(scene, action, ignore_collisions, scene.robot.arm, scene.robot.gripper)
        done = False
        while not done:
            done = path.step()
            scene.step()
            if self._callable_each_step is not None:
                self._callable_each_step(scene.get_observation())
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success:
                break    

    def get_path(self, scene: Scene, action: np.ndarray, ignore_collisions: bool, arm: Arm, gripper: Gripper):
        if not self._absolute_mode and self._frame != 'end effector':
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == 'world' else arm.get_tip()
        self._quick_boundary_check(scene, action)

        colliding_shapes = []
        if not ignore_collisions:
            if self._robot_shapes is None:
                self._robot_shapes = arm.get_objects_in_tree(
                    object_type=ObjectType.SHAPE)
            # First check if we are colliding with anything
            colliding = arm.check_arm_collision()
            if colliding:
                # Disable collisions with the objects that we are colliding with
                grasped_objects = gripper.get_grasped_objects()
                colliding_shapes = [
                    s for s in scene.pyrep.get_objects_in_tree(
                        object_type = ObjectType.SHAPE) if (
                            s.is_collidable() and
                            s not in self._robot_shapes and
                            s not in grasped_objects and
                            arm.check_arm_collision(
                                s))]
                [s.set_collidable(False) for s in colliding_shapes]

        try:
            # try once with collision checking (if ignore_collisions is true)
            try:
                path = arm.get_path(
                    action[:3],
                    quaternion=action[3:],
                    ignore_collisions=ignore_collisions,
                    relative_to=relative_to,
                    trials=200, #..TODO was 100
                    max_configs=10, #..TODO was 10
                    max_time_ms=20, #..TODO was 10
                    trials_per_goal=10, #..TODO was 5
                    algorithm=Algos.RRTConnect
                )
                return path
            except ConfigurationPathError as e:
                if ignore_collisions:
                    raise InvalidActionError(
                        'A path could not be found. Most likely due to the target '
                        'being inaccessible or a collison was detected.') from e
                else:
                    # try once more with collision checking disabled
                    path = arm.get_path(
                        action[:3],
                        quaternion=action[3:],
                        ignore_collisions=True,
                        relative_to=relative_to,
                        trials=100,
                        max_configs=10,
                        max_time_ms=10,
                        trials_per_goal=5,
                        algorithm=Algos.RRTConnect
                    )
        except ConfigurationPathError as e:
            raise InvalidActionError(
                'A path could not be found. Most likely due to the target '
                'being inaccessible or a collison was detected.') from e


    def action_shape(self, scene: Scene) -> tuple:
        return 7,



class UnimanualEndEffectorPoseViaPlanning(EndEffectorPoseViaPlanning):

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: str = 'world',
                 collision_checking: bool = False,
                 robot_name: str = ''):
        super().__init__(absolute_mode, frame, collision_checking)
        self.robot_name = robot_name

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if self.robot_name == 'right':
            path = self.get_path(scene, action, ignore_collisions, scene.robot.right_arm, scene.robot.right_gripper)
        elif self.robot_name == 'left':
            path = self.get_path(scene, action, ignore_collisions, scene.robot.left_arm, scene.robot.left_gripper)
        else:
            logging.error('Invalid robot name')

        if not path:
            logging.warning('No path found')
            return
        done = False
        while not done:
            done = path.step()
            scene.step()
            if self._callable_each_step is not None:
                # Record observations
                self._callable_each_step(scene.get_observation())
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success and self._callable_each_step is None:
                break

class BimanualEndEffectorPoseViaPlanning(EndEffectorPoseViaPlanning):


    def action(self, scene: Scene, action: np.ndarray, ignore_collisions):

        assert_action_shape(action, self.action_shape(scene))
 
        right_action = action[:7]
        left_action = action[7:]

        right_ignore_collision = ignore_collisions[0]
        left_ignore_collison = ignore_collisions[1]

        assert_unit_quaternion(right_action[3:])
        assert_unit_quaternion(left_action[3:])

        right_done = True
        left_done = True
        try:
            right_path = self.get_path(scene, right_action, right_ignore_collision, scene.robot.right_arm, scene.robot.right_gripper)
            if right_path:
                right_done = False
            else:
                logging.warning("right path is none")
        except (ConfigurationPathError, InvalidActionError):
            pass
        
        try:
            left_path = self.get_path(scene, left_action, left_ignore_collison, scene.robot.left_arm, scene.robot.left_gripper)
            if left_path:
                left_done = False
            else:
                logging.warning("left path is none")
        except (ConfigurationPathError, InvalidActionError):
            pass
        

        done = False

        limit_time = 5 # seconds
        duration = 0
        start_time = time.time()
        while not done:
            if not right_done and right_path:
                right_done = right_path.step()
            if not left_done and left_path:
                left_done = left_path.step()

            done = right_done and left_done
            scene.step()
            if self._callable_each_step is not None:
                if not isinstance(self._callable_each_step, tuple):
                    self._callable_each_step(scene.get_observation())
                else:
                    self._callable_each_step[0](scene.get_observation())
                    self._callable_each_step[1](scene.get_observation())

            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success:
                break
            end_time = time.time()
            duration = end_time - start_time
            if duration > limit_time:
                logging.warning(f"Path execution time {duration} exceeded limit of {limit_time} seconds, breaking out of loop.")
                break
        if self._callable_each_step is not None and isinstance(self._callable_each_step, tuple):
            self._callable_each_step[0](scene.get_observation(), is_key_frame=True)


    def action_shape(self, scene: Scene) -> tuple:
        return 14,

    def unimanual_action_shape(self, scene: Scene) -> tuple:
        return 7,


class EndEffectorPoseViaIK(ArmActionMode):
    """High-level action where target pose is given and reached via IK.

    Given a target pose, IK via inverse Jacobian is performed. This requires
    the target pose to be close to the current pose, otherwise the action
    will fail. It is up to the user to constrain the action to
    meaningful values.

    The decision to apply collision checking is a crucial trade off!
    With collision checking enabled, you are guaranteed collision free paths,
    but this may not be applicable for task that do require some collision.
    E.g. using this mode on pushing object will mean that the generated
    path will actively avoid not pushing the object.
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: str = 'world',
                 collision_checking: bool = False):
        """
        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either 'world' or 'end effector'.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        if frame not in ['world', 'end effector']:
            raise ValueError(
                "Expected frame to one of: 'world, 'end effector'")

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != 'end effector':
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == 'world' else scene.robot.arm.get_tip()

        try:
            joint_positions = scene.robot.arm.solve_ik_via_jacobian(
                action[:3], quaternion=action[3:], relative_to=relative_to)
            scene.robot.arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            raise InvalidActionError(
                'Could not perform IK via Jacobian; most likely due to current '
                'end-effector pose being too far from the given target pose. '
                'Try limiting/bounding your action space.') from e
        done = False
        prev_values = None
        # Move until reached target joint positions or until we stop moving
        # (e.g. when we collide wth something)
        while not done:
            scene.step()
            cur_positions = scene.robot.arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=0.01)
            not_moving = False
            if prev_values is not None:
                not_moving = np.allclose(
                    cur_positions, prev_values, atol=0.001)
            prev_values = cur_positions
            done = reached or not_moving

    def action_shape(self, scene: Scene) -> tuple:
        return 7,


class BimanualEndEffectorPoseViaIKAdvanced(EndEffectorPoseViaIK, EndEffectorPoseViaPlanning):
    
    def action(self, scene: Scene, action: np.ndarray, ignore_collisions):
        assert_action_shape(action, self.action_shape(scene))

        right_action = action[:7]
        left_action = action[7:]

        right_ignore_collision = ignore_collisions[0]
        left_ignore_collison = ignore_collisions[1]

        assert_unit_quaternion(right_action[3:])
        assert_unit_quaternion(left_action[3:])

        if not self._absolute_mode and self._frame != 'end effector':
            right_action = calculate_delta_pose(scene.robot.right_arm, right_action)
            left_action = calculate_delta_pose(scene.robot.left_arm, left_action)

        relative_to_right = None if self._frame == 'world' else scene.robot.right_arm.get_tip()
        relative_to_left = None if self._frame == 'world' else scene.robot.left_arm.get_tip()

        right_ik_success = False
        left_ik_success = False
        try:
            right_joint_positions = scene.robot.right_arm.solve_ik_via_jacobian(
                right_action[:3], quaternion=right_action[3:], relative_to=relative_to_right)
            right_ik_success = True
            scene.robot.right_arm.set_joint_target_positions(right_joint_positions)
        except IKError as e:
            # raise InvalidActionError(
            #     'Could not perform IK via Jacobian; most likely due to current '
            #     'right end-effector pose being too far from the given target pose. '
            #     'Try limiting/bounding your action space.') from e
            
            print("right ik failed, trying using EndEffectorPoseViaPlanning")
            right_path = self.get_path(scene, right_action, right_ignore_collision, scene.robot.right_arm, scene.robot.right_gripper)
            if right_path:
                right_done = False


        
        try:
            left_joint_positions = scene.robot.left_arm.solve_ik_via_jacobian(
                left_action[:3], quaternion=left_action[3:], relative_to=relative_to_left)
            left_ik_success = True
            scene.robot.left_arm.set_joint_target_positions(left_joint_positions)
        except IKError as e:
            # raise InvalidActionError(
            #     'Could not perform IK via Jacobian; most likely due to current '
            #     'left end-effector pose being too far from the given target pose. '
            #     'Try limiting/bounding your action space.') from e
            print("left ik failed, trying using EndEffectorPoseViaPlanning")
            left_path = self.get_path(scene, left_action, left_ignore_collison, scene.robot.left_arm, scene.robot.left_gripper)
            if left_path:
                left_done = False

        done = False
        prev_right_values = None
        prev_left_values = None
        limit_time = 2 # seconds
        duration = 0
        # Move until reached target joint positions or until we stop moving
        # (e.g. when we collide wth something)
        start_time = time.time()
        while not done:
            if right_ik_success and left_ik_success:
                scene.step()

                if self._callable_each_step is not None:
                    self._callable_each_step(scene.get_observation())

                cur_right_positions = scene.robot.right_arm.get_joint_positions()
                cur_left_positions = scene.robot.left_arm.get_joint_positions()

                right_reached = np.allclose(cur_right_positions, right_joint_positions, atol=0.01)
                left_reached = np.allclose(cur_left_positions, left_joint_positions, atol=0.01)

                not_moving = False
                if prev_right_values is not None and prev_left_values is not None:
                    not_moving = (np.allclose(
                        cur_right_positions, prev_right_values, atol=0.001) and
                        np.allclose(cur_left_positions, prev_left_values, atol=0.001))
                prev_right_values = cur_right_positions
                prev_left_values = cur_left_positions

                done = (right_reached and left_reached) or not_moving

            elif right_ik_success and not left_ik_success:
                if not left_done:
                    left_done = left_path.step()
                cur_right_positions = scene.robot.right_arm.get_joint_positions()
                right_reached = np.allclose(cur_right_positions, right_joint_positions, atol=0.01)
                done = right_reached and left_done
                scene.step()
                if self._callable_each_step is not None:
                    self._callable_each_step(scene.get_observation())

            elif not right_ik_success and left_ik_success:
                if not right_done:
                    right_done = right_path.step()
                cur_left_positions = scene.robot.left_arm.get_joint_positions()
                left_reached = np.allclose(cur_left_positions, left_joint_positions, atol=0.01)
                done = left_reached and right_done
                scene.step()
                if self._callable_each_step is not None:
                    self._callable_each_step(scene.get_observation())

            else:
                if not right_done:
                    right_done = right_path.step()
                if not left_done:
                    left_done = left_path.step()
                done = right_done and left_done
                scene.step()
                if self._callable_each_step is not None:
                    self._callable_each_step(scene.get_observation())
            
            end_time = time.time()
            duration = end_time - start_time
            if duration > limit_time:
                print("Time limit exceeded while trying to reach target pose via IK or planning. Breaking out of loop.")
                break
            
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success:
                break
                
    def action_shape(self, scene: Scene) -> tuple:
        return 14,

    def unimanual_action_shape(self, scene: Scene) -> tuple:
        return 7,


class BimanualOSC(ArmActionMode):
    """Operational Space Control (OSC) for bimanual arms.
    
    Implements impedance-based Cartesian control in task space.
    Inspired by Khatib 1987 OSC framework, but adapted to RLBench 
    constraints (Jacobian-only, no mass matrix inversion).
    
    Key improvements over simple IK:
    - Natural compliance via impedance control (stiffness + damping)
    - Smooth velocity feedback instead of on/off control
    - Proper nullspace handling for redundant DOFs
    - Flexible reference frames (world/ee) and action types (absolute/delta)
    """

    TORQUE_MAX_VEL = 9999
    TORQUE_ZERO_EPS = 1e-8

    def __init__(self,
                 input_type: str = 'absolute',
                 frame: str = 'world',
                 disable_gravity: bool = True,
                 kp_pos: float = 1000.0,       # 更加柔顺的位置刚度
                 kp_rot: float = 500.0,        # 更加柔顺的姿态刚度
                 damping_ratio_pos: float = 1.0,  # 适中阻尼
                 damping_ratio_rot: float = 1.0,
                 nullspace_gain: float = 0.05,
                 nullspace_damping: float = 0.01,
                 damped_inv_lambda: float = 0.2,
                 max_torque: float = 50.,    # 极大降低最大力矩，保证绝对安全
                 desired_task_velocity: np.ndarray = None):
        """
        Args:
            input_type (str): 'absolute' or 'delta' action interpretation
            frame (str): 'world' or 'ee' (end-effector) reference frame
            disable_gravity (bool): Whether to disable global gravity using pyrep simConst.
            kp_pos (float): Proportional gain for position [N/m]
            kp_rot (float): Proportional gain for rotation [N·m/rad]
            damping_ratio_pos (float): Damping ratio for position (1.0 = critical damping)
            damping_ratio_rot (float): Damping ratio for rotation
            nullspace_gain (float): Gain for nullspace control
            nullspace_damping (float): Nullspace control damping
            damped_inv_lambda (float): Damping parameter for Jacobian pseudo-inverse
            max_torque (float): Torque clipping limit [N·m]
            desired_task_velocity (np.ndarray): Optional desired task-space velocity [6,]
        """
        assert input_type in ['delta', 'absolute'], \
            f"input_type must be 'delta' or 'absolute', got {input_type}"
        assert frame in ['world', 'ee'], \
            f"frame must be 'world' or 'ee', got {frame}"
        
        self._input_type = input_type
        self._frame = frame
        self._disable_gravity = disable_gravity
        
        # Impedance parameters (following robosuite convention)
        self._kp_pos = kp_pos
        self._kp_rot = kp_rot
        # Derivative gains computed from kp and damping ratio
        # kd = 2 * sqrt(kp) * damping_ratio
        self._kd_pos = 2.0 * np.sqrt(kp_pos) * damping_ratio_pos
        self._kd_rot = 2.0 * np.sqrt(kp_rot) * damping_ratio_rot
        
        # Nullspace control parameters
        self._nullspace_gain = nullspace_gain
        self._nullspace_damping = nullspace_damping
        
        # Jacobian damping (for pseudo-inverse stability)
        self._damped_inv_lambda = damped_inv_lambda
        self._max_torque = max_torque
        
        # Optional desired task velocity for feedforward
        if desired_task_velocity is None:
            self._desired_task_velocity = np.zeros(6)
        else:
            self._desired_task_velocity = np.asarray(desired_task_velocity, dtype=float)
            if self._desired_task_velocity.shape != (6,):
                raise ValueError('Expected desired_task_velocity to have shape (6,).')
        
        # Cache rest poses for nullspace control
        self._right_q_rest = None
        self._left_q_rest = None

        self.step = 0
        
        logging.info(f"BimanualOSC initialized: input_type={input_type}, frame={frame}, "
                    f"kp_pos={kp_pos}, kp_rot={kp_rot}")

    @staticmethod
    def _compose_delta_pose(current_pose: np.ndarray,
                            delta_action: np.ndarray,
                            frame: str = 'world') -> np.ndarray:
        """Compose target pose from current pose and delta action.
        
        Args:
            current_pose (np.ndarray): Current [x, y, z, qx, qy, qz, qw]
            delta_action (np.ndarray): Delta [dx, dy, dz, dqx, dqy, dqz, dqw]
            frame (str): 'world' - position delta in world frame
                        'ee' - position delta in end-effector frame
        
        Returns:
            np.ndarray: Target pose [x, y, z, qx, qy, qz, qw]
        """
        c_x, c_y, c_z, c_qx, c_qy, c_qz, c_qw = current_pose
        d_x, d_y, d_z, d_qx, d_qy, d_qz, d_qw = delta_action

        current_q = Quaternion(c_qw, c_qx, c_qy, c_qz)
        
        # If frame is 'ee', rotate position delta into world frame
        if frame == 'ee':
            d_x, d_y, d_z = current_q.rotate([d_x, d_y, d_z])
        
        # Compose rotation: apply delta rotation, then current rotation
        new_q = Quaternion(d_qw, d_qx, d_qy, d_qz) * current_q
        n_qw, n_qx, n_qy, n_qz = list(new_q)
        
        return np.array([
            c_x + d_x,
            c_y + d_y,
            c_z + d_z,
            n_qx,
            n_qy,
            n_qz,
            n_qw,
        ])

    @staticmethod
    def _quaternion_error(target_quat_xyzw: np.ndarray,
                          current_quat_xyzw: np.ndarray) -> np.ndarray:
        """Compute orientation error in Euler angles (small angle approximation).
        
        For small angles, the error quaternion can be converted to rotation vector 
        which approximates Euler angles.
        
        Args:
            target_quat_xyzw (np.ndarray): Target quaternion [qx, qy, qz, qw]
            current_quat_xyzw (np.ndarray): Current quaternion [qx, qy, qz, qw]
        
        Returns:
            np.ndarray: Orientation error [roll, pitch, yaw] in radians
        """
        t_qx, t_qy, t_qz, t_qw = target_quat_xyzw
        c_qx, c_qy, c_qz, c_qw = current_quat_xyzw

        q_target = Quaternion(t_qw, t_qx, t_qy, t_qz)
        q_current = Quaternion(c_qw, c_qx, c_qy, c_qz)
        # Error quaternion: how much to rotate from current to target
        q_err = (q_target * q_current.inverse).normalised
        
        # Convert to Euler angles for intuitive error representation
        yaw, pitch, roll = q_err.yaw_pitch_roll
        return np.array([roll, pitch, yaw])

    def _compute_target_pose(self, arm: Arm, action: np.ndarray) -> np.ndarray:
        """Compute target pose from action based on input_type and frame.
        
        Args:
            arm (Arm): Robot arm object
            action (np.ndarray): Action [x, y, z, qx, qy, qz, qw]
        
        Returns:
            np.ndarray: Target pose [x, y, z, qx, qy, qz, qw]
        """
        current_pose = np.array(arm.get_tip().get_pose())
        
        if self._input_type == 'absolute':
            # Absolute mode: action is the target pose directly
            return np.array(action)
        
        # Delta mode: compose target from current pose and delta action
        # Input frame: 'world' or 'ee' (end-effector)
        return self._compose_delta_pose(current_pose, action, frame=self._frame)

    def _osc_step(self,
                  hand: str,
                  arm: Arm,
                  target_pose: np.ndarray,
                  q_rest: np.ndarray) -> np.ndarray:
        """Execute one OSC control step via impedance control.
        
        Computes joint torques to track target pose with impedance control:
        - Task-space PD control: F = kp * e_pos + kd * e_vel
        - Jacobian mapping to joint space with damping
        - Nullspace control for maintaining initial pose
        
        Args:
            arm (Arm): Robot arm object
            target_pose (np.ndarray): Target pose [x, y, z, qx, qy, qz, qw]
            q_rest (np.ndarray): Rest joint positions for nullspace control
        
        Returns:
            np.ndarray: Joint torque commands
        """
        # Current state
        q = np.array(arm.get_joint_positions())
        dq = np.array(arm.get_joint_velocities())
        tip_pose = np.array(arm.get_tip().get_pose())
        tip_vel = np.array(arm.get_tip().get_velocity()).flatten()  # [vx, vy, vz, wx, wy, wz]

        from pyrep.const import ObjectType
        link_shapes = arm.get_objects_in_tree(object_type=ObjectType.SHAPE)

        for shape in link_shapes:
            # 物理引擎中只有被设置为 dynamic (动态支持) 的物体才参与重力和碰撞的质量计算
            if shape.is_dynamic():
                # print(f"部件 {shape.get_name()} 的质量为: {shape.get_mass()} kg")
                if shape.get_name() == "Panda_rightArm_gripper":
                    right_gripper_pose = shape.get_pose()
                    right_gripper_vel = shape.get_velocity()
                if shape.get_name() == "Panda_leftArm_gripper":
                    left_gripper_pose = shape.get_pose()
                    left_gripper_vel = shape.get_velocity()

        # if hand == 'right':
        #     tip_pose = right_gripper_pose
        # else:
        #     tip_pose = left_gripper_pose
        # Task-space errors
        # pos_err = target_pose[:3] - tip_pose[:3]  # desired - actual [m]
        # rot_err = self._quaternion_error(target_pose[3:], tip_pose[3:])  # [rad]
        min_ee_pos_error = -0.2
        max_ee_pos_error = 0.2
        pos_err = np.clip(target_pose[:3] - tip_pose[:3], min_ee_pos_error, max_ee_pos_error)
        min_ee_rot_error =  -np.deg2rad(30)
        max_ee_rot_error =  np.deg2rad(30)
        rot_err = np.clip(self._quaternion_error(target_pose[3:], tip_pose[3:]), min_ee_rot_error, max_ee_rot_error)

        task_err = np.concatenate([pos_err, rot_err], axis=0)  # [6,]
        
        # Apply error deadband to prevent numerical instability at small errors
        # When error is very small, numerical precision can cause control to go in reverse
        pos_deadband = 1e-4  # 0.1mm
        rot_deadband = 1e-4  # rad
        task_err[:3] = np.where(np.abs(task_err[:3]) < pos_deadband, 0, task_err[:3])
        task_err[3:] = np.where(np.abs(task_err[3:]) < rot_deadband, 0, task_err[3:])
        
        # DEBUG: Check Jacobian shape
        jac_T_flipped = arm.get_jacobian()
        # jac_T_raw = np.array(arm.get_jacobian())
        jac_T_raw = np.flipud(jac_T_flipped)
        # jac_raw = np.flipud(jac_raw)
        # # Jacobian from PyRep: typically [6 x 7] for translational + rotational (6D task space)
        # # But user says it returns [7x6], so let's handle both cases
        # if jac_raw.shape == (7, 6):
        #     # If shape is [7x6], treat it as [N_dof x 6_partials], so transpose to [6x7]
        #     jacobian = jac_raw.T  # [6 x 7]
        # elif jac_raw.shape == (6, 7):
        #     # If already [6x7], use directly
        #     jacobian = jac_raw
        # else:
        #     raise ValueError(f'Unexpected Jacobian shape {jac_raw.shape}; expected (6, 7) or (7, 6).')
        
        # jacobian[5, :] *= -1
        # # Task-space velocity via forward kinematics (Jacobian * joint_velocity)
        # task_vel = jacobian @ qd  # [6,]
        task_vel = tip_vel
        
        # Impedance control gains
        task_kp = np.diag([
            self._kp_pos, self._kp_pos, self._kp_pos,      # position gains
            self._kp_rot, self._kp_rot, self._kp_rot,       # rotation gains
        ])
        task_kd = np.diag([
            self._kd_pos, self._kd_pos, self._kd_pos,       # position damping
            self._kd_rot, self._kd_rot, self._kd_rot,        # rotation damping
        ])
        
        # Desired wrench computed from impedance law:
        # F = kp * e_task + kd * (v_desired - v_current)
        # vel_err = self._desired_task_velocity - task_vel
        vel_err = jac_T_raw.T @ dq
        print(task_err, vel_err)
        # desired_wrench = np.multiply(task_kp, task_err) + 0 * np.multiply(task_kd, vel_err)  # [6,]
        desired_wrench = task_kp @ task_err + task_kd @ vel_err  # [6,]
        
        # Clamp wrench to prevent torque saturation for large errors
        # When wrench is too large, all torques hit the limit and arm can't move effectively
        # Limit position force to ~500N and torque to ~100 N·m for stability
        max_force = 500.0  # [N]
        max_torque_wrench = 100.0  # [N·m]
        
        pos_force_norm = np.linalg.norm(desired_wrench[:3])
        rot_torque_norm = np.linalg.norm(desired_wrench[3:])
        
        if pos_force_norm > max_force:
            desired_wrench[:3] *= max_force / pos_force_norm
        if rot_torque_norm > max_torque_wrench:
            desired_wrench[3:] *= max_torque_wrench / rot_torque_norm
        
        # --- Map task wrench to joint torques ---
        # Use simple Jacobian transpose (most stable for OSC when Jacobian is well-scaled)
        # tau = J.T @ F_desired
        tau_task = jac_T_raw @ desired_wrench  # [7,]
        
        # --- Nullspace control (DISABLED for debugging) ---
        # Temporarily disabled to isolate task-space control issues
        # tau_null = 0
        
        # Total torque command (task-space only for now)
        tau_cmd = tau_task  # Bypass nullspace projection for debugging
        
        # Clip to max torque
        tau_cmd = np.clip(tau_cmd, -self._max_torque, self._max_torque)
        # tau_cmd = np.zeros_like(tau_cmd)  # DEBUG: Disable torques to test if arm can reach target pose via IK alone
        # tau_cmd[1:2] = 0.1
        print(tau_cmd)

        
        return tau_cmd

    def _apply_joint_torques(self, arm: Arm, tau: np.ndarray):
        # tml = BimanualOSC.TORQUE_MAX_VEL
        # target_velocities = []
        # if not isinstance(tau, list):
        #     tau = tau.tolist()
        # for t in tau:
        #     if abs(t) <= BimanualOSC.TORQUE_ZERO_EPS:
        #         target_velocities.append(0.0)
        #     else:
        #         target_velocities.append(tml if t < 0.0 else -tml)
        # # target_velocities = np.zeros_like(target_velocities)  # Override to zero for debugging
        # arm.set_joint_target_velocities(target_velocities)
        # arm.set_joint_forces(np.abs(tau).tolist())
        arm.set_joint_forces(tau.tolist())

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        """Execute OSC for both arms.
        
        Args:
            scene (Scene): RLBench scene
            action (np.ndarray): Shape [14,] = [right_pose (7), left_pose (7)]
                where each pose is [x, y, z, qx, qy, qz, qw]
        """
        assert_action_shape(action, self.action_shape(scene))

        if self._disable_gravity:
            from pyrep.backend import sim
            gravity_ptr = sim.ffi.new('float[3]', [0.0, 0.0, 0.0])
            sim.simSetArrayParameter(sim.sim_arrayparam_gravity, gravity_ptr)
            
        from pyrep.backend import sim
        sim.simSetFloatParameter(sim.sim_floatparam_simulation_time_step, 0.001)  # 1000 Hz simulation step

        from pyrep.const import ObjectType

        # 假设 scene.robot.right_arm 或者你的 arm 对象已经定义
        # 提取机械臂层级树下的所有形状(Shape)
        link_shapes = scene.robot.right_arm.get_objects_in_tree(object_type=ObjectType.SHAPE)

        for shape in link_shapes:
            # 物理引擎中只有被设置为 dynamic (动态支持) 的物体才参与重力和碰撞的质量计算
            if shape.is_dynamic():
                print(f"部件 {shape.get_name()} 的质量为: {shape.get_mass()} kg")

        right_action = np.array(action[:7])
        left_action = np.array(action[7:])


        # scene.robot.right_arm.set_joint_target_velocities(np.zeros_like(right_action))
        # scene.robot.left_arm.set_joint_target_velocities(np.zeros_like(left_action))
        
        # Validate quaternions in action
        assert_unit_quaternion(right_action[3:])
        assert_unit_quaternion(left_action[3:])

        # Initialize rest poses on first call
        if self._right_q_rest is None:
            self._right_q_rest = np.array(scene.robot.right_arm.get_joint_positions())
        if self._left_q_rest is None:
            self._left_q_rest = np.array(scene.robot.left_arm.get_joint_positions())

        # Compute target poses based on input_type and frame
        right_target_pose = self._compute_target_pose(scene.robot.right_arm, right_action)
        left_target_pose = self._compute_target_pose(scene.robot.left_arm, left_action)

        # Execute OSC and get joint torques
        right_tau = self._osc_step("right", scene.robot.right_arm, right_target_pose, self._right_q_rest)
        left_tau = self._osc_step("left", scene.robot.left_arm, left_target_pose, self._left_q_rest)

        # Apply torques to both arms
        self._apply_joint_torques(scene.robot.right_arm, right_tau)
        self._apply_joint_torques(scene.robot.left_arm, left_tau)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        # right_action = np.array(action[:7])
        # right_vel = np.zeros_like(right_action)
        # # right_vel[1:2] = 20
        # left_action = np.array(action[7:])
        # left_vel = np.zeros_like(left_action)
        # # left_vel[1:2] = 20
        # # self._apply_joint_torques(scene.robot.right_arm, scene.robot.right_arm.get_joint_forces())
        # scene.robot.right_arm.set_joint_target_velocities(right_vel)
        # # self._apply_joint_torques(scene.robot.left_arm, scene.robot.left_arm.get_joint_forces())
        # scene.robot.left_arm.set_joint_target_velocities(left_vel)
        pass

    def action_step(self, scene: Scene):
        # if self.step == 0:
        # right_action = np.array([0]*7)
        # right_vel = np.zeros_like(right_action)
        # # right_vel[1:2] = 20
        # left_action = np.array([0]*7)
        # left_vel = np.zeros_like(left_action)
        # # left_vel[1:2] = 20
        # # self._apply_joint_torques(scene.robot.right_arm, scene.robot.right_arm.get_joint_forces())
        # scene.robot.right_arm.set_joint_target_velocities(right_vel)
        # # self._apply_joint_torques(scene.robot.left_arm, scene.robot.left_arm.get_joint_forces())
        # scene.robot.left_arm.set_joint_target_velocities(left_vel)
        scene.step()
        self.step += 1
        print(scene.robot.right_arm.get_joint_velocities())
        if self._callable_each_step is not None:
            self._callable_each_step(scene.get_observation())




    def action_shape(self, scene: Scene) -> tuple:
        return 14,

    def unimanual_action_shape(self, scene: Scene) -> tuple:
        return 7,

    def set_control_mode(self, robot: Robot):
        # OSC applies wrench/torque-level commands, so disable internal
        # position control loops that would otherwise hold joints.
        robot.right_arm.set_control_loop_enabled(False)
        robot.left_arm.set_control_loop_enabled(False)
        # robot.right_arm.set_motor_locked_at_zero_velocity(False)
        # robot.left_arm.set_motor_locked_at_zero_velocity(False)
    
    def get_config_dict(self) -> dict:
        """Return configuration for logging/serialization.
        
        Useful for recording which OSC parameters were used in experiments.
        """
        return {
            'input_type': self._input_type,
            'frame': self._frame,
            'kp_pos': self._kp_pos,
            'kp_rot': self._kp_rot,
            'kd_pos': self._kd_pos,
            'kd_rot': self._kd_rot,
            'nullspace_gain': self._nullspace_gain,
            'nullspace_damping': self._nullspace_damping,
            'damped_inv_lambda': self._damped_inv_lambda,
            'max_torque': self._max_torque,
        }

    # def action_pre_step(self, scene: Scene, action: np.ndarray):

    #     if self._disable_gravity:
    #         from pyrep.backend import sim
    #         gravity_ptr = sim.ffi.new('float[3]', [0.0, 0.0, 0.0])
    #         sim.simSetArrayParameter(sim.sim_arrayparam_gravity, gravity_ptr)
            
    #     from pyrep.backend import sim
    #     sim.simSetFloatParameter(sim.sim_floatparam_simulation_time_step, 0.001)  # 1000 Hz simulation step

    #     assert_action_shape(action, self.action_shape(scene))
    #     right_action = action[:7]
        
    #     right_action = np.zeros_like(right_action)  # Override to zero for debugging
    #     right_action[0:1] = 10
    #     left_action = action[7:]
    #     left_action = np.zeros_like(left_action)  # Override to zero for debugging
    #     scene.robot.right_arm.set_joint_target_velocities(right_action)
    #     scene.robot.left_arm.set_joint_target_velocities(left_action)

    # def action_post_step(self, scene: Scene, action: np.ndarray):
    #     # scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))
    #     # Current state
    #     q = np.array(scene.robot.right_arm.get_joint_positions())
    #     qd = np.array(scene.robot.right_arm.get_joint_velocities())
    #     tip_pose = np.array(scene.robot.right_arm.get_tip().get_pose())
    #     tip_vel = np.array(scene.robot.right_arm.get_tip().get_velocity()).flatten()  # [vx, vy, vz, wx, wy, wz]

    #     from pyrep.const import ObjectType
    #     link_shapes = scene.robot.right_arm.get_objects_in_tree(object_type=ObjectType.SHAPE)

    #     for shape in link_shapes:
    #         # 物理引擎中只有被设置为 dynamic (动态支持) 的物体才参与重力和碰撞的质量计算
    #         if shape.is_dynamic():
    #             # print(f"部件 {shape.get_name()} 的质量为: {shape.get_mass()} kg")
    #             if shape.get_name() == "Panda_rightArm_gripper":
    #                 right_gripper_pose = shape.get_pose()
    #                 right_gripper_vel = shape.get_velocity()
    #             if shape.get_name() == "Panda_leftArm_gripper":
    #                 left_gripper_pose = shape.get_pose()
    #                 left_gripper_vel = shape.get_velocity()


    #     # DEBUG: Check Jacobian shape
    #     jac_raw = np.array(scene.robot.right_arm.get_jacobian())
    #     # jacobian = jac_raw.flatten().reshape((7, 6), order='C')  # Force column-major reshape to get correct orientation
    #     jac_raw = np.flipud(jac_raw)
    #     # Jacobian from PyRep: typically [6 x 7] for translational + rotational (6D task space)
    #     # But user says it returns [7x6], so let's handle both cases
    #     if jac_raw.shape == (7, 6):
    #         # If shape is [7x6], treat it as [N_dof x 6_partials], so transpose to [6x7]
    #         jacobian = jac_raw.T  # [6 x 7]
    #     elif jac_raw.shape == (6, 7):
    #         # If already [6x7], use directly
    #         jacobian = jac_raw
    #     else:
    #         raise ValueError(f'Unexpected Jacobian shape {jac_raw.shape}; expected (6, 7) or (7, 6).')
        
    #     jacobian[5, :] *= -1
    #     # Task-space velocity via forward kinematics (Jacobian * joint_velocity)
    #     task_vel = jacobian @ qd  # [6,]
    #     # task_vel = tip_vel
        


    #     right_action = action[:7]
    #     left_action = action[7:]
    #     scene.robot.right_arm.set_joint_target_velocities(np.zeros_like(right_action))
    #     scene.robot.left_arm.set_joint_target_velocities(np.zeros_like(left_action))

    # def action_shape(self, scene: Scene) -> tuple:
    #     return SUPPORTED_ROBOTS[scene.robot_setup][2],

    # def set_control_mode(self, robot: Robot):
    #     robot.right_arm.set_control_loop_enabled(False)
    #     robot.right_arm.set_motor_locked_at_zero_velocity(True)
    #     robot.left_arm.set_control_loop_enabled(False)
    #     robot.left_arm.set_motor_locked_at_zero_velocity(True)
