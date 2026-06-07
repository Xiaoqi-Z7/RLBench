from typing import List
from pyrep.objects.joint import Joint
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task
from rlbench.backend.task import BimanualTask
from collections import defaultdict

from pyrep.objects.shape import Shape
from rlbench.backend.conditions import Condition
import numpy as np

class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False



class BimanualPickLaptop(BimanualTask):

    def init_task(self) -> None:
        self.register_success_conditions([LiftedCondition(Shape('lid'), 1.0)])
        self.waypoint_mapping = defaultdict(lambda: 'left')
        for i in range(1, 7, 2):
            self.waypoint_mapping.update({f'waypoint{i}': 'right'})

    def init_episode(self, index: int) -> List[str]:
        return ['pick up the laptop']

    def variation_count(self) -> int:
        return 1
    
    def get_obj_poses(self):
        poses = {}
        for obj in self.task_relevant_objects():
            poses[obj.get_name()] = obj.get_pose()
        return poses
    
    def task_relevant_objects(self):
        return [Shape('lid'), Shape('base')]
    
    def get_task_relevant_obj_count(self):
        return len(self.task_relevant_objects())

    def get_existing_relevant_objs_mask(self):
        mask = np.ones(self.get_task_relevant_obj_count(), dtype=bool)
        return mask