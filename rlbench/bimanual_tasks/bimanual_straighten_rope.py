from typing import List


from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task
from rlbench.backend.task import BimanualTask
from collections import defaultdict

import numpy as np

class BimanualStraightenRope(BimanualTask):

    def init_task(self) -> None:
        self.register_success_conditions(
            [DetectedCondition(Shape('head'), ProximitySensor('success_head')),
             DetectedCondition(Shape('tail'), ProximitySensor('success_tail'))])

        self.waypoint_mapping = defaultdict(lambda: 'right')
        for i in range(3):
            self.waypoint_mapping[f'waypoint{i}'] = 'left'

    def init_episode(self, index: int) -> List[str]:
        return ['straighten rope',
                'pull the rope straight',
                'grasping each end of the rope in turn, leave the rope straight'
                ' on the table',
                'pull each end of the rope until is is straight',
                'tighten the rope',
                'pull the rope tight']

    def variation_count(self) -> int:
        return 1
    
    # def get_obj_poses(self):
    #     poses = {}
    #     for obj in self.task_relevant_objects():
    #         poses[obj.get_name()] = obj.get_pose()
    #     return poses
    
    def task_relevant_objects(self):
        return [Shape('head'), Shape('tail'), Shape('success_head'), Shape('success_tail')]
    
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
        for obj_name in ['head', 'tail', 'success_head', 'success_tail']:
            GlobalRegistryGenerator.force_register(task_name, obj_name, color_rgb=None)
            
        print(f"✅ Successfully pre-registered objects for task: {task_name}")
