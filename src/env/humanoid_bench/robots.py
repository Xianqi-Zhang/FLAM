class H1:
    dof = 26
    action_space_dim = 19
    # * Camera_names related camera_name in h1_pos.xml.
    camera_names = [  # * 10.
        "cam_kitchen", "cam_default", "cam_maze", "cam_tabletop", "cam_hurdle",
        "cam_basketball", "cam_hand_visible", "cam_inhand", "left_eye_camera", "right_eye_camera"
    ]

    def __init__(self, env=None):
        self._env = env

    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self._env.named.data.xmat["torso_link", "zz"]

    def head_height(self):
        """Returns the height of the torso."""
        return self._env.named.data.site_xpos["head", "z"]

    def left_foot_height(self):
        """Returns the height of the left foot."""
        return self._env.named.data.site_xpos["left_foot", "z"]

    def right_foot_height(self):
        """Returns the height of the right foot."""
        return self._env.named.data.site_xpos["right_foot", "z"]

    def center_of_mass_position(self):
        """Returns position of the center-of-mass."""
        return self._env.named.data.subtree_com["pelvis"].copy()

    def center_of_mass_velocity(self):
        """Returns the velocity of the center-of-mass."""
        return self._env.named.data.sensordata["pelvis_subtreelinvel"].copy()

    def body_velocity(self):
        """Returns the velocity of the torso in local frame."""
        return self._env.named.data.sensordata["body_velocimeter"].copy()

    def torso_vertical_orientation(self):
        """Returns the z-projection of the torso orientation matrix."""
        return self._env.named.data.xmat["torso_link", ["zx", "zy", "zz"]]

    # * -------------------------------------------------
    # * Added.
    def camera_pose(self):
        current_cam_index = self.camera_names.index(self._env.camera_name)
        current_cam_orient = self._env.data.cam_xmat[current_cam_index]
        current_cam_pos = self._env.data.cam_xpos[current_cam_index]
        return current_cam_orient, current_cam_pos

    def global_position(self):
        # * Shape: (71, 3)
        # print(self._env.named.data.xpos)
        return self._env.data.xpos[:71, :].copy()  # * Only select robot part.

    def global_orientation(self):
        # * Shape: (71, 4)
        # print(self._env.named.data.xquat)
        return self._env.data.xquat[:71, :].copy()  # * Only select robot part.

    # * -------------------------------------------------
    def joint_angles(self):
        """Returns the state without global orientation or position."""
        # Skip the 7 DoFs of the free root joint.
        return self._env.data.qpos[7: self.dof].copy()

    def joint_velocities(self):
        """Returns the joint velocities."""
        return self._env.data.qvel[6: self.dof - 1].copy()

    def control(self):
        """Returns a copy of the control signals for the actuators."""
        return self._env.data.ctrl.copy()

    def actuator_forces(self):
        """Returns a copy of the forces applied by the actuators."""
        return self._env.data.actuator_force.copy()

    def left_hand_position(self):
        return self._env.named.data.site_xpos["left_hand"]

    def left_hand_velocity(self):
        return self._env.named.data.sensordata["left_hand_subtreelinvel"].copy()

    def left_hand_orientation(self):
        return self._env.named.data.site_xmat["left_hand"]

    def right_hand_position(self):
        return self._env.named.data.site_xpos["right_hand"]

    def right_hand_velocity(self):
        return self._env.named.data.sensordata["right_hand_subtreelinvel"].copy()

    def right_hand_orientation(self):
        return self._env.named.data.site_xmat["right_hand"]


class H1Hand(H1):
    dof = 76
    action_space_dim = 61


class H1SimpleHand(H1):
    dof = 52
    action_space_dim = 61


class H1Touch(H1):
    dof = 76
    action_space_dim = 61


class H1Strong(H1):
    dof = 76
    action_space_dim = 61


class G1(H1):
    dof = 44
    action_space_dim = 61
