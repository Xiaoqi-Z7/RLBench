from typing import List

import numpy

from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import BimanualTask
from collections import defaultdict
from rlbench.backend.spawn_boundary import SpawnBoundary

import numpy as np


DIRT_NUM = 5


class BimanualSweepToDustpan(BimanualTask):

    def init_task(self) -> None:
        success_sensor = ProximitySensor('success')
        self.dirts = [Shape('dirt' + str(i)) for i in range(DIRT_NUM)]
        conditions = [DetectedCondition(dirt, success_sensor) for dirt in self.dirts]
        self.register_graspable_objects([Shape('broom'), Shape('Dustpan_4'),  Shape('Dustpan_5'), Shape('Dustpan_3')])
        self.register_success_conditions(conditions)
        self.waypoint_mapping = defaultdict(lambda: 'left')
        self.waypoint_mapping.update({'waypoint5': 'right', 'waypoint6': 'right', 'waypoint7': 'right'})

    def init_episode(self, index: int) -> List[str]:

        b = SpawnBoundary([Shape('dirt_boundary')])
        b.clear()
        for item in self.dirts:
            b.sample(item, min_distance=0.01, ignore_collisions=True)

        return ['sweep dirt to dustpan',
                'sweep the dirt up',
                'use the broom to brush the dirt into the dustpan',
                'clean up the dirt',
                'pick up the brush and clean up the table',
                'grasping the broom by its handle, clear way the dirt from the '
                'table',
                'leave the table clean']

    def variation_count(self) -> int:
        return 1
    
    def is_static_workspace(self):
        return True
    
    # def get_obj_poses(self):
    #     poses = {}
    #     for obj in self.task_relevant_objects():
    #         poses[obj.get_name()] = obj.get_pose()
    #     return poses
    
    def task_relevant_objects(self):
        return [Shape('broom'), Shape('Dustpan_4'),  Shape('Dustpan_5'), Shape('Dustpan_3')] + self.dirts
    
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
        for obj_name in ['broom', 'Dustpan_4', 'Dustpan_5', 'Dustpan_3'] + [f'dirt{i}' for i in range(DIRT_NUM)]:
            GlobalRegistryGenerator.force_register(task_name, obj_name, color_rgb=None)
            
        print(f"✅ Successfully pre-registered objects for task: {task_name}")
