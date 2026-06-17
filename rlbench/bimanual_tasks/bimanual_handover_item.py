from collections import defaultdict
from typing import List, Tuple

import numpy as np

from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import BimanualTask
from rlbench.backend.spawn_boundary import SpawnBoundary
from pyrep.objects.dummy import Dummy
from rlbench.backend.exceptions import BoundaryError
from absl import logging
from pyrep.objects.object import Object
from rlbench.backend.conditions import Condition

import re


colors = [
    ('red', (1.0, 0.0, 0.0)),
    ('green', (0.0, 1.0, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('yellow', (1.0, 1.0, 0.0)),
    #('olive', (0.5, 0.5, 0.0)),
    ('purple', (0.5, 0.0, 0.5)),
    #('teal', (0, 0.5, 0.5)),
    #('black', (0.0, 0.0, 0.0)),
    #('white', (1.0, 1.0, 1.0)),
]

class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False

class BimanualHandoverItem(BimanualTask):

    def init_task(self) -> None:

        self.items = [Shape(f'item{i}') for i in range(5)]

        self.register_graspable_objects(self.items)

        self.waypoint_mapping = defaultdict(lambda: 'left')
        self.waypoint_mapping.update({'waypoint0': 'right', 'waypoint5': 'right'})

        self.boundaries = Shape('handover_item_boundary')
        self.boundary_init_pose = Shape('handover_item_boundary').get_position()


    def init_episode(self, index:  int) -> List[str]:

        self._variation_index = index

        color_name, color = colors[index]
        self.items[0].set_color(color)

        remaining_colors = colors.copy()
        remaining_colors.remove((color_name, color))
        np.random.shuffle(remaining_colors)

        for i, item in enumerate(self.items[1:]):
            item.set_color(remaining_colors[i][1])

        Shape('handover_item_boundary').set_position(self.boundary_init_pose)

        b = SpawnBoundary([self.boundaries])
        b.MAX_SAMPLES = 1000
        b.clear()
        for item in self.items:
            try:
                b.sample(item, min_distance=0.05)
            except BoundaryError as err:
                logging.warning("error %s. Sampling again while ignoring collisions ", err)
                b.sample(item, ignore_collisions=True, min_distance=0.05)

        right_success_sensor = ProximitySensor('Panda_rightArm_gripper_attachProxSensor')
        left_success_sensor = ProximitySensor('Panda_leftArm_gripper_attachProxSensor')

        self.register_success_conditions(
            [DetectedCondition(self.items[0], right_success_sensor),  
             DetectedCondition(self.items[0], left_success_sensor, negated=True),
             LiftedCondition(self.items[0], 0.8)])

        return [f'bring me the {color_name} item',
                f'hand over the {color_name} object']

    def variation_count(self) -> int:
        return len(colors)
    
    #def boundary_root(self) -> Object:
    #    return Shape('handover_item_boundary')

    def is_static_workspace(self):
        return True

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
    
    # def get_obj_poses(self):
    #     poses = {}
    #     for color in colors:
    #         for obj in self.task_relevant_objects():
    #             if obj.get_color() == color[1]:
    #                 poses[obj.get_name()] = obj.get_pose()
    #     return poses
    
    def task_relevant_objects(self):
        return self.items
    
    # def get_task_relevant_obj_count(self):
    #     return len(self.task_relevant_objects())
    
    # def get_existing_relevant_objs_mask(self):
    #     mask = np.ones(self.get_task_relevant_obj_count(), dtype=bool)
    #     return mask
    @classmethod
    def pre_register_all(cls):
        """
        [STATIC PRE-REGISTRATION]
        Run once at startup to register all possible object-color combinations 
        for this task into the global registry.
        """
        from rlbench.bimanual_tasks.generate_registry import GlobalRegistryGenerator
        task_name = cls.__name__
        
        # 优化后：所有的物理方块统称为 'item'。
        # 既然一共只有 5 种颜色，我们只需要注册 5 个语义 ID 即可。
        for _, rgb_val in colors:
            GlobalRegistryGenerator.force_register(task_name, 'item', rgb_val)
                
        print(f"✅ Successfully pre-registered 5 combinations for task: {task_name}")

    def get_obj_poses(self):
        """
        Returns two separate dictionaries containing object poses and metadata:
        - local_poses: Compressed, contiguous integer IDs starting from 0 (Single-Task).
        - global_poses: Absolute static integer IDs from the master registry (Multi-Task).
        - num_local_type: Size of the local subset vocabulary.
        - num_global_type: Size of the total global vocabulary.
        - num_objects: Number of relevant physical objects currently in the scene.
        """
        from rlbench.bimanual_tasks.generate_registry import LocalRegistryReader
        local_poses = {}
        global_poses = {}
        task_name = self.__class__.__name__ 
        
        # Automatically initialize and compute vocabulary sizes on the first call
        if not hasattr(self, '_local_id_translator'):
            # Load the single-task compact mapping
            self._local_id_translator = LocalRegistryReader.get_task_subset_mapping(task_name)
            self._total_local_objects = len(self._local_id_translator)
            
            # AUTOMATIC SOLUTION: The total number of unique IDs in the entire 
            # global JSON file is exactly the total global vocabulary size.
            LocalRegistryReader.load() # Ensure file is read
            self._total_global_objects = len(LocalRegistryReader._mapping)
            
            print(f"📊 [Vocabulary Info] Task: {task_name} | Local Size: {self._total_local_objects} | Global Total Size: {self._total_global_objects}")

        num_global_type = self._total_global_objects
            
        for obj in self.task_relevant_objects():
            raw_name = obj.get_name() 
            
            # [CRITICAL FIX] Strip digits from the end to unify 'item0', 'item1', etc., into 'item'
            base_name = re.sub(r'\d+$', '', raw_name)
            
            # Safe color RGB extraction (only applied for tasks that randomize colors)
            color_rgb = None
            if task_name in ['BimanualDualPushButtons', 'BimanualHandoverItem']:
                if hasattr(obj, 'get_color'):
                    try:
                        color_rgb = obj.get_color()
                        if color_rgb is not None:
                            color_rgb = tuple(color_rgb)
                    except Exception:
                        color_rgb = None
            
            # 1. Fetch the absolute global ID using the cleaned base_name!
            global_id = LocalRegistryReader.get_id(task_name, base_name, color_rgb)
            
            # 2. Translate to the compact local ID
            local_id = self._local_id_translator[global_id]
            
            # 3. Populate both data dictionaries
            global_poses[global_id] = obj.get_pose()
            local_poses[local_id] = obj.get_pose()

        num_local_type = self._total_local_objects
        num_objects = len(self.task_relevant_objects())
        
        return local_poses, global_poses, num_local_type, num_global_type, num_objects
