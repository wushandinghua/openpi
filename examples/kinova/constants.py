### Task parameters
# This is the reset position that is used by the standard kinova runtime.
DEFAULT_RESET_POSITION = [-0.05114174767722801, -0.32670062356438123, -3.067645192233781, -1.9892705467878438, 0.01766729831947718, -1.091837990020724, 1.7992638567067234]

### Kinova fixed constants
DT = 0.001

# max width of robotiq gripper
GRIPPER_WIDTH_MAX = 0.085
GRIPPER_JOINT_MAX, GRIPPER_JOINT_MIN = 0.79301, 0.00698

CAM_WRIST = "cam_wrist"
CAM_EXTERIOR = "cam_exterior"

############################ Helper functions ############################

GRIPPER_POSITION_NORMALIZE_FN = lambda x: 1 - x / GRIPPER_WIDTH_MAX
GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: (1 - x) *  GRIPPER_WIDTH_MAX
)
