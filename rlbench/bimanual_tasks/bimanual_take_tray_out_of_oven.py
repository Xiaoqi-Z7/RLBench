from collections import defaultdict
from typing import List, Tuple


from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.robot import BimanualRobot
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from rlbench.backend.task import BimanualTask

import numpy as np

class BimanualTakeTrayOutOfOven(BimanualTask):

    def init_task(self) -> None:
        success_detector = ProximitySensor('success')
        tray = Shape('tray')
        assert(isinstance(self.robot, BimanualRobot))
        self.register_graspable_objects([tray])
        self.register_success_conditions(
            [DetectedCondition(tray, success_detector, negated=True),
             NothingGrasped(self.robot.right_gripper), NothingGrasped(self.robot.left_gripper)])

        self.waypoint_mapping = defaultdict(lambda: 'left')
        for i in range(4):
            self.waypoint_mapping[f'waypoint{i}'] = 'right'

    def init_episode(self, index: int) -> List[str]:
        return ['take tray out of oven',
                'open the oven and take the baking tray out',
                'grasp the handle on the over door, open it, and remove the '
                'tray from the oven',
                'get the baking tray from the oven',
                'get the tray',
                'take out the tray',
                'retrieve the oven tray']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -3.14 / 4.], [0, 0, 3.14 / 4.]

    def boundary_root(self) -> Object:
        return Shape('oven_boundary_root')

    # def get_obj_poses(self):
    #     poses = {}
    #     for obj in self.task_relevant_objects():
    #         poses[obj.get_name()] = obj.get_pose()
    #     return poses
    
    def task_relevant_objects(self):
        return [Shape('tray'), Shape('oven_base'), Shape('oven_door')]

    # def get_task_relevant_obj_count(self):
    #     return len(self.task_relevant_objects())

    # def get_existing_relevant_objs_mask(self):
    #     mask = np.ones(self.get_task_relevant_obj_count(), dtype=bool)
    #     return mask
    @classmethod
    def pre_register_all(cls):
        """
        [STATIC PRE-REGISTRATION]
        Run once at startup to register all possible objects 
        for this task into the global registry.
        """
        from rlbench.bimanual_tasks.generate_registry import GlobalRegistryGenerator
        task_name = cls.__name__
        
        # This task only has one relevant object: the 'ball'.
        # Since it does not randomize colors, color_rgb is None.
        for obj_name in ['tray', 'oven_base', 'oven_door']:
            GlobalRegistryGenerator.force_register(task_name, obj_name, color_rgb=None)
            
        print(f"✅ Successfully pre-registered objects for task: {task_name}")