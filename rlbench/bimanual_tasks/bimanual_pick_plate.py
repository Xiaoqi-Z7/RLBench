from typing import List, Tuple


from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import BimanualTask
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from collections import defaultdict


from rlbench.backend.conditions import Condition
import numpy as np

class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False



class BimanualPickPlate(BimanualTask):

    def init_task(self) -> None:
        self.plate = Shape('plate')
      
        self.register_success_conditions([ LiftedCondition(self.plate, 1.0)])
        self.register_graspable_objects([self.plate])
        self.waypoint_mapping = defaultdict(lambda: 'right')
        self.waypoint_mapping.update({'waypoint0': 'left', 'waypoint2': 'left', 'waypoint6': 'left'})

    def init_episode(self, index: int) -> List[str]:
        return ['pick up the plate']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def get_obj_poses(self):
        poses = {}
        poses['plate'] = self.plate.get_pose()
        return poses
    
    # def get_obj_poses(self):
    #     poses = {}
    #     for obj in self.task_relevant_objects():
    #         poses[obj.get_name()] = obj.get_pose()
    #     return poses
    
    def task_relevant_objects(self):
        return [Shape('plate')]
    
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
        for obj_name in ['plate']:
            GlobalRegistryGenerator.force_register(task_name, obj_name, color_rgb=None)
            
        print(f"✅ Successfully pre-registered objects for task: {task_name}")