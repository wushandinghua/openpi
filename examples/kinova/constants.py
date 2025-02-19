### Task parameters
# This is the reset position that is used by the standard kinova runtime.
DEFAULT_RESET_POSITION = [1.1717908796348647e-05, -0.35005974211768365, 3.1400381664615136, -2.54007670871461, -7.084008499624872e-05, -0.8700355533690383, 1.5699754073888796]

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
