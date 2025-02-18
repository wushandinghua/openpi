### Task parameters
# This is the reset position that is used by the standard kinova runtime.
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]

### Kinova fixed constants
DT = 0.001

# max width of robotiq gripper
GRIPPER_POSITION_OPEN = 0.05800

CAM_WRIST = "cam_wrist"
CAM_EXTERIOR = "cam_exterior"

############################ Helper functions ############################

GRIPPER_POSITION_NORMALIZE_FN = lambda x: 1 - x / GRIPPER_POSITION_OPEN
GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: (1 - x) *  GRIPPER_POSITION_OPEN
)
