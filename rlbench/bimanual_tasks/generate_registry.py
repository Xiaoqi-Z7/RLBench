import json
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
# Import your task classes
from rlbench.bimanual_tasks.bimanual_dual_push_buttons import BimanualDualPushButtons # Replace with your actual task paths

class GlobalRegistryGenerator:
    _registry = {}
    _counter = 0

    @classmethod
    def force_register(cls, task_name: str, obj_name: str, color_rgb: tuple = None):
        if color_rgb is not None:
            rgb_str = f"({color_rgb[0]:.2f}, {color_rgb[1]:.2f}, {color_rgb[2]:.2f})"
            semantic_key = f"{obj_name}:{rgb_str}"
        else:
            semantic_key = obj_name
            
        full_key = f"{task_name}@{semantic_key}" # Use '@' as a separator for easy serialization
        
        if full_key not in cls._registry:
            # Store the auto-incremented integer ID
            cls._registry[full_key] = cls._counter
            cls._counter += 1

    @classmethod
    def save_to_json(cls, filepath="global_id_registry.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cls._registry, f, indent=4, ensure_ascii=False)
        print(f"Static registry successfully saved to: {filepath} with a total of {cls._counter} object keys.")


class LocalRegistryReader:
    _mapping = None

    @classmethod
    def load(cls, filepath="global_id_registry.json"):
        if cls._mapping is None:
            with open(filepath, "r") as f:
                cls._mapping = json.load(f)

    @classmethod
    def get_id(cls, task_name: str, obj_name: str, color_rgb: tuple = None) -> int:
        """Used for Multi-Task Training: Returns the global absolute ID."""
        cls.load()
        if color_rgb is not None:
            rgb_str = f"({color_rgb[0]:.2f}, {color_rgb[1]:.2f}, {color_rgb[2]:.2f})"
            semantic_key = f"{obj_name}:{rgb_str}"
        else:
            semantic_key = obj_name
            
        full_key = f"{task_name}@{semantic_key}"
        if full_key not in cls._mapping:
            raise ValueError(f"Key '{full_key}' not found in the static registry!")
        return cls._mapping[full_key]

    @classmethod
    def get_task_subset_mapping(cls, task_name: str) -> dict:
        """
        Used for Single-Task Training: Filters out all possible objects of this specific task,
        and re-maps them to a compact, contiguous local ID range starting from 0.
        Returns a dictionary: { global_id: local_compact_id }
        """
        cls.load()
        
        # 1. Extract all global IDs belonging to this task
        task_global_ids = []
        for key, global_id in cls._mapping.items():
            if key.startswith(f"{task_name}@"):
                task_global_ids.append(global_id)
                
        # 2. Sort to guarantee absolute consistency across runs
        task_global_ids.sort()
        
        # 3. Create a re-indexed compact mapping: {Global ID -> Local ID (0, 1, 2...)}
        subset_mapping = {global_id: local_id for local_id, global_id in enumerate(task_global_ids)}
        return subset_mapping


if __name__ == "__main__":
    # 1. Must initialize an RLBench environment (this launches the backend PyRep simulator)
    # Otherwise, importing or using the Task class directly will fail due to a missing simulator handle.
    action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
    env = Environment(action_mode, headless=True) # headless=True runs without popping up the 3D GUI for faster speed
    env.launch()

    # 2. List all the task classes you need to support
    all_task_classes = [
        BimanualDualPushButtons,
        # BimanualOpenDoor, 
        # Other tasks...
    ]

    print("Scanning all tasks and pre-registering all potential objects...")
    
    # 3. Trigger static pre-registration for each task
    for task_cls in all_task_classes:
        # To be safe, you can instantiate the task via the environment first to ensure 
        # internal variables (like `colors`) are correctly loaded, though `pre_register_all` is a classmethod.
        if hasattr(task_cls, 'pre_register_all'):
            # Call the pre_register_all method you implemented in the task class
            # Internally, it will call GlobalRegistryGenerator.force_register
            task_cls.pre_register_all() 

    # 4. Shutdown the simulator
    env.shutdown()

    # 5. Save the generated mapping to the local disk
    GlobalRegistryGenerator.save_to_json()