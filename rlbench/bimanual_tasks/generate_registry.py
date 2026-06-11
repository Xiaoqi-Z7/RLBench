import json
import os
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
# Import your task classes
from rlbench.bimanual_tasks.bimanual_dual_push_buttons import BimanualDualPushButtons
from rlbench.bimanual_tasks.bimanual_handover_item_easy import BimanualHandoverItemEasy
from rlbench.bimanual_tasks.bimanual_handover_item import BimanualHandoverItem
from rlbench.bimanual_tasks.bimanual_lift_ball import BimanualLiftBall
from rlbench.bimanual_tasks.bimanual_lift_tray import BimanualLiftTray
from rlbench.bimanual_tasks.bimanual_pick_laptop import BimanualPickLaptop
from rlbench.bimanual_tasks.bimanual_pick_plate import BimanualPickPlate
from rlbench.bimanual_tasks.bimanual_push_box import BimanualPushBox
from rlbench.bimanual_tasks.bimanual_put_bottle_in_fridge import BimanualPutBottleInFridge
from rlbench.bimanual_tasks.bimanual_put_item_in_drawer import BimanualPutItemInDrawer
from rlbench.bimanual_tasks.bimanual_straighten_rope import BimanualStraightenRope
from rlbench.bimanual_tasks.bimanual_sweep_to_dustpan import BimanualSweepToDustpan
from rlbench.bimanual_tasks.bimanual_take_tray_out_of_oven import BimanualTakeTrayOutOfOven

REGISTRY_FILEPATH = "/home/xiaoqi/master_thesis/bil/robomimic_rlbench/data_instruction/global_id_registry.json"

class GlobalRegistryGenerator:
    _registry = {}
    _counter = 0
    _loaded = False

    @classmethod
    def _load_existing_registry(cls, filepath=REGISTRY_FILEPATH):
        """Load existing registry if it exists, to preserve previous registrations."""
        if cls._loaded:
            return
        
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    cls._registry = json.load(f)
                    # Recalculate counter based on existing entries
                    if cls._registry:
                        cls._counter = max(cls._registry.values()) + 1
                    cls._loaded = True
                    print(f"  Loaded existing registry with {len(cls._registry)} entries")
            except Exception as e:
                print(f"  Warning: Could not load existing registry: {e}")
                cls._registry = {}
                cls._counter = 0
        else:
            cls._loaded = True

    @classmethod
    def force_register(cls, task_name: str, obj_name: str, color_rgb: tuple = None):
        # Ensure existing registry is loaded first
        cls._load_existing_registry()
        
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
    def save_to_json(cls, filepath=REGISTRY_FILEPATH):
        # Ensure existing registry is loaded first
        cls._load_existing_registry(filepath)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cls._registry, f, indent=4, ensure_ascii=False)
        print(f"Static registry successfully saved to: {filepath} with a total of {len(cls._registry)} object keys.")


class LocalRegistryReader:
    _mapping = None

    @classmethod
    def load(cls, filepath=REGISTRY_FILEPATH):
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
    # IMPORT from the module system to ensure the __main__ script shares the exact same 
    # GlobalRegistryGenerator class identity (and its _registry dictionary) as the task classes!
    from rlbench.bimanual_tasks.generate_registry import GlobalRegistryGenerator as ActiveGenerator

    # 0. Load existing registry if it exists (to preserve previous registrations)
    ActiveGenerator._load_existing_registry()
    
    # 1. Must initialize an RLBench environment (this launches the backend PyRep simulator)
    # Otherwise, importing or using the Task class directly will fail due to a missing simulator handle.
    action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
    env = Environment(action_mode, headless=True) # headless=True runs without popping up the 3D GUI for faster speed
    env.launch()

    # 2. List all the task classes you need to support
    all_task_classes = [
        BimanualDualPushButtons,
        BimanualHandoverItemEasy,
        BimanualHandoverItem,
        BimanualLiftBall,
        BimanualLiftTray,
        BimanualPickLaptop,
        BimanualPickPlate,
        BimanualPushBox,
        BimanualPutBottleInFridge,
        BimanualPutItemInDrawer,
        BimanualStraightenRope,
        BimanualSweepToDustpan,
        BimanualTakeTrayOutOfOven,
        # Other tasks...
    ]

    print("Scanning all tasks and pre-registering all potential objects...")
    
    # 3. Trigger static pre-registration for each task
    registered_count = 0
    for task_cls in all_task_classes:
        # To be safe, you can instantiate the task via the environment first to ensure 
        # internal variables (like `colors`) are correctly loaded, though `pre_register_all` is a classmethod.
        if hasattr(task_cls, 'pre_register_all'):
            # Call the pre_register_all method you implemented in the task class
            # Internally, it will call GlobalRegistryGenerator.force_register
            try:
                task_cls.pre_register_all()
                print(f"  ✓ Registered objects for task: {task_cls.__name__}")
                registered_count += 1
            except Exception as e:
                print(f"  ✗ Error registering {task_cls.__name__}: {e}")
        else:
            print(f"  ⚠ Task {task_cls.__name__} does not have 'pre_register_all' method")
    
    print(f"\nSuccessfully processed {registered_count}/{len(all_task_classes)} tasks")
    
    if len(ActiveGenerator._registry) == 0:
        print("WARNING: Registry is empty! No objects were registered.")
    else:
        print(f"Registry contains {len(ActiveGenerator._registry)} entries") 

    # 4. Shutdown the simulator
    env.shutdown()

    # 5. Save the generated mapping to the local disk
    if len(ActiveGenerator._registry) > 0:
        ActiveGenerator.save_to_json()
        print(f"\n✓ Registry saved successfully with {len(ActiveGenerator._registry)} entries!")
    else:
        print("\n✗ ERROR: Registry is empty. Not saving to avoid overwriting existing data.")
        print("   Please check that:")
        print("   - Task classes have 'pre_register_all()' method")
        print("   - pre_register_all() calls ActiveGenerator.force_register()")
        print("   - Task environment is properly initialized")