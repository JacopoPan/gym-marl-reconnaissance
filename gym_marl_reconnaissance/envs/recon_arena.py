"""Multi-agent reinforcement learning reconnaissance environment.

"""
import os
import gym
import numpy as np
import pybullet as p
import pybullet_data

from datetime import datetime
from enum import Enum
from PIL import Image


class ActionType(str, Enum):
    """Enumeration class for different implementations of the action space.

    """

    TASK_ASSIGNMENT = 'task_assignment'
    TRACKING = 'tracking'


class ObsType(str, Enum):
    """Enumeration class for different implementations of the observations space.

    """

    GLOBAL = 'global'


class RewardChoice(str, Enum):
    """Enumeration class for different implementations of the reward signal.

    """

    REWARD_A = 'reward_a'


class AdversaryType(str, Enum):
    """Enumeration class for different implementations of scripted behavior of non-learning agents.

    """

    BLIND = 'blind'
    AVOIDANT = 'avoidant'


class ReconArena(gym.Env):
    """Reconnaissance environment.

    Attributes
    ----------
    CTRL_FREQ : int
        The control---each call to .step()---frequency of the environment in Hz.
    PYB_FREQ : int
        The stepping frequency of PyBullet in Hz.
    EPISODE_LENGTH_SEC : int
        The episode duration in seconds.
    action_space : gym.spaces
        The action space of the environment.
    observation_space : gym.spaces
        The observation space of the environment.

    """

    def __init__(self,
                 seed: int = 1337,
                 ctrl_freq: int = 5,
                 pyb_freq: int = 60,
                 gui: bool = False,
                 record: bool = False,
                 episode_length_sec: int = 30,
                 action_type: ActionType = ActionType.TASK_ASSIGNMENT,
                 obs_type: ObsType = ObsType.GLOBAL,
                 reward_choice: RewardChoice = RewardChoice.REWARD_A,
                 adv_type: AdversaryType = AdversaryType.AVOIDANT,
                 visibility_threshold: int = 8,
                 setup: dict = {'edge':10, 'obstacles':0, 'tt':1, 's1':1, 'adv':2, 'neu':1},
                 debug: bool = False
                 ):
        """__init__ method.

        Parameters
        ----------
        param1 : str
            Description of `param1`.
        param2 : :obj:`list` of :obj:`str`
            Description of `param2`. Multiple
            lines are supported.
        param3 : :obj:`int`, optional
            Description of `param3`.

        seed : int, optional
            The randomization seed.
        ctrl_freq : int, optional
            The control frequency of the environment in Hz.
        pyb_freq : int, optional
            The stepping frequency of PyBullet in Hz.
        gui : bool, optional
            Whether to display PyBullet's GUI.
        record : bool, optional
            Whether to record a video of the environment.
        episode_length_sec : int, optional
            The episode duration in seconds.
        action_type : ActionType, optional
            The action space of the environment.
        obs_type : ObsType, optional
            The observation space of the environment.
        reward_choice : RewardChoice, optional
            The reward function of the environment.
        adv_type : AdversaryType, optional
            The type of scripted behavior of the non-learning adversaries.
        visibility_threshold : int, optional
            The visibility range of the learning agents, in meters.
        setup : dict, optional
            The edge of the arena, in meters, the number of obstacles,
            tellos, robomasters, adversaries and neutral agents.
        debug : bool, optional
            Whether to draw debug information on PyBullet's GUI.

        """
        np.random.seed(seed)
        super(ReconArena, self).__init__()
        # Setup timing attributes.
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        if (self.PYB_FREQ % self.CTRL_FREQ != 0) or (self.PYB_FREQ < self.CTRL_FREQ):
            raise ValueError('[ERROR] in ReconArena.__init__(), pyb_freq is not divisible by ctrl_freq.')
        self.PYB_UPDATES_PER_STEP = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ
        self.EPISODE_LENGTH_SEC = episode_length_sec
        # Setup physical attributes.
        self.GRAVITY_ACC = 9.8
        self.TT_MASS = 0.08
        # Setup environment's (and agent's) attributes.
        self.VISIBILITY_THRESHOLD = visibility_threshold
        self.ACTION_TYPE = action_type
        self.OBS_TYPE = obs_type
        self.REWARD_CHOICE = reward_choice
        self.ADV_TYPE = adv_type
        self.ARENA_EDGE = setup['edge']
        self.NUM_OBSTACLES = setup['obstacles']
        self.NUM_TT = setup['tt']
        self.NUM_S1 = setup['s1']
        self.NUM_ADVERSARY = setup['adv']
        self.NUM_NEUTRAL = setup['neu']
        if self.ACTION_TYPE == ActionType.TASK_ASSIGNMENT:
            self.NUM_OBSTACLES = 0
            self.VISIBILITY_THRESHOLD = np.inf
            print('[WARNING] ...')
        # Setup GUI attributes.
        self.GUI = gui
        self.RECORD = record
        self.RENDER_WIDTH = int(640)
        self.RENDER_HEIGHT = int(480)
        self.PYB_CLIENT = -1
        if self.GUI:
            self.PYB_CLIENT = p.connect(p.GUI)
            # for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
            #     p.configureDebugVisualizer(i, 0, physicsClientId=self.PYB_CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=1.5*setup['edge'],
                                         cameraYaw=315,
                                         cameraPitch=-50,
                                         cameraTargetPosition=[setup['edge']/2, setup['edge']/2, 0],
                                         physicsClientId=self.PYB_CLIENT)
        else:
            self.PYB_CLIENT = p.connect(p.DIRECT)
        self.VIEW_MATRIX = p.computeViewMatrixFromYawPitchRoll(distance=1.5*setup['edge'],
                                                               yaw=315,
                                                               pitch=-50,
                                                               cameraTargetPosition=[setup['edge']/2, setup['edge']/2, 0],
                                                               roll=0,
                                                               upAxisIndex=2,
                                                               physicsClientId=self.PYB_CLIENT)
        self.PROJ_MATRIX = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.RENDER_WIDTH) / self.RENDER_HEIGHT,
                                                        nearVal=0.1,
                                                        farVal=100.0,
                                                        physicsClientId=self.PYB_CLIENT)
        FRAME_PER_SEC = 30
        if (self.PYB_FREQ%FRAME_PER_SEC) !=0 or (self.PYB_FREQ < FRAME_PER_SEC):
            raise ValueError('')
        self.CAPTURE_FREQ = int(self.PYB_FREQ/FRAME_PER_SEC)
        self.DEBUG = debug
        # Make initial reset mandatory.
        self.initial_reset = False
        # Set up action and observation spaces.
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space()

    def reset(self):
        """Reset the environment between episodes.

        Returns
        -------
        ndarray
            The initial observation.

        """
        # Reset counters.
        self.initial_reset = True
        self.pyb_step_counter = 0
        self.ctrl_step_counter = 0
        # Set PyBullet options.
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.PYB_CLIENT)
        # Load URDF models.
        p.loadURDF('plane.urdf', physicsClientId=self.PYB_CLIENT)
        self.OBSTACLES_IDS = self._create_obstacles()
        self.TT_IDS = self._create_team_tt()
        self.S1_IDS = self._create_team_s1()
        self.ADVERSARY_IDS = self._create_adversary()
        self.NEUTRAL_IDS = self._create_neutral()
        # Reset scripted, non-learning agents behaviors.
        self.adversary_directions = np.ones(self.NUM_ADVERSARY)
        self.ADVERSARY_SPEEDS = np.random.uniform(low=0.5, high=1.5, size=(self.NUM_ADVERSARY))
        self.neutral_directions = np.ones(self.NUM_NEUTRAL)
        self.NEUTRAL_SPEEDS = np.random.uniform(low=0.5, high=1.5, size=(self.NUM_NEUTRAL))
        # Create ids for the debug information (to avoid rendering flickering).
        if self.DEBUG:
            self.OBSTACLES_DEBUG_LABELS = -1*np.ones(self.NUM_OBSTACLES)
            self.TT_DEBUG_LABELS = -1*np.ones(self.NUM_TT)
            self.S1_DEBUG_LABELS = -1*np.ones(self.NUM_S1)
            self.ADVERSARY_DEBUG_LABELS = -1*np.ones(self.NUM_ADVERSARY)
            self.NEUTRAL_DEBUG_LABELS = -1*np.ones(self.NUM_NEUTRAL)
        # Start recording the episode (as .mp4).
        if self.RECORD and self.GUI:
            self.VIDEO_ID = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
                                                fileName=os.path.dirname(os.path.abspath(__file__))+'/../../saves/video-'+datetime.now().strftime('%m.%d.%Y_%H.%M.%S')+'.mp4',
                                                physicsClientId=self.PYB_CLIENT
                                                )
        # Create a folder to save the video as .png frames.
        if self.RECORD and not self.GUI:
            self.frame_num = 0
            self.IMG_PATH = os.path.dirname(os.path.abspath(__file__))+'/../../saves/video-'+datetime.now().strftime('%m.%d.%Y_%H.%M.%S')+'/'
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)
        # Update internal state.
        self._update_env_state()
        # Return initial observation.
        obs = self._compute_obs()
        return obs

    def step(self,
             action
             ):
        """Step the environment.

        Parameters
        ----------
        action : ndarray
            The action selected by a controller, learning agent.

        Returns
        -------
        ndarray
            The observation at the end of the step.
        float
            The reward signal at the end of the step.
        bool
            Whether the episode ends at the current step.
        dict
            Additional information (unused).

        """
        # Check te initial reset happened.
        if not self.initial_reset:
            raise RuntimeError('[ERROR] You must call ReconArena.reset() at least once before using ReconArena.step().')
        #
        # Set PyBullet velocities.
        self._apply_action(action)
        self._nonstationary_adversary()
        self._nonstationary_neutral()
        # Step the PyBullet engine
        for _ in range(self.PYB_UPDATES_PER_STEP):
            # Record a frame if saving the episode as .pngs.
            if self.RECORD and not self.GUI and self.pyb_step_counter%self.CAPTURE_FREQ == 0:
                [w, h, rgb, dep, seg] = p.getCameraImage(width=self.RENDER_WIDTH,
                                                         height=self.RENDER_HEIGHT,
                                                         shadow=1,
                                                         viewMatrix=self.VIEW_MATRIX,
                                                         projectionMatrix=self.PROJ_MATRIX,
                                                         renderer=p.ER_TINY_RENDERER,
                                                         flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                         physicsClientId=self.PYB_CLIENT
                                                         )
                (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(self.IMG_PATH+'frame_'+str(self.frame_num)+'.png')
                self.frame_num += 1
            # Lift tellos from the burden of gravity.
            for i in range(self.NUM_TT):
                p.applyExternalForce(objectUniqueId=self.TT_IDS[i], linkIndex=-1,
                                     forceObj=np.array([0, 0, self.GRAVITY_ACC]) * self.TT_MASS,
                                     posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.PYB_CLIENT)
            p.stepSimulation(physicsClientId=self.PYB_CLIENT)
            self.pyb_step_counter += 1
        # Update the internal state of the environment.
        self._update_env_state()  # Note: where the update takes place in .step() matters.
        # Compute observation, reward, and done.
        obs = self._compute_obs()
        reward = self._compute_reward()
        done = self._compute_done()
        self.ctrl_step_counter += 1
        # Add debug information to the GUI.
        if self.DEBUG:
            self._debug_ids()
        return obs, reward, done, {}

    def render(self):
        """Render the environment.

        Returns
        -------
        ndarray
            An RGB frame captured at the current step of the environment.

        """
        if self.PYB_CLIENT >= 0:
            (w, h, rgb, _, _) = p.getCameraImage(width=self.RENDER_WIDTH,
                                                 height=self.RENDER_HEIGHT,
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                                 viewMatrix=self.VIEW_MATRIX,
                                                 projectionMatrix=self.PROJ_MATRIX,
                                                 physicsClientId=self.PYB_CLIENT)
        # Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA').show()
        return np.reshape(rgb, (h, w, 4))

    def close(self):
        """Close the environment.

        """
        if self.PYB_CLIENT >= 0:
            if self.RECORD and self.GUI:
                p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.PYB_CLIENT)
            p.disconnect(physicsClientId=self.PYB_CLIENT)
        self.PYB_CLIENT = -1

    def _set_action_space(self):
        """Create the action space the environment.

        Returns
        -------
        gym.spaces
            The action space.

        """
        NUM_AGENTS = self.NUM_TT + self.NUM_S1
        if self.ACTION_TYPE == ActionType.TASK_ASSIGNMENT:
            NUM_OPTIONS = self.NUM_ADVERSARY + self.NUM_NEUTRAL  # As many as the assignable tasks.
            return gym.spaces.MultiDiscrete(np.repeat(NUM_OPTIONS, NUM_AGENTS))
        elif self.ACTION_TYPE == ActionType.TRACKING:
            NUM_OPTIONS = 5  # Stay, forward, backward, left, right.
            return gym.spaces.MultiDiscrete(np.repeat(NUM_OPTIONS, NUM_AGENTS))
        else:
            raise ValueError('')

    def _set_observation_space(self):
        """Create the observation space of the environment.

        Returns
        -------
        gym.spaces
            The observation space.

        """
        if self.OBS_TYPE == ObsType.GLOBAL:
            OBS_SIZE = 5 * (self.NUM_TT + self.NUM_S1 + self.NUM_ADVERSARY + self.NUM_NEUTRAL + self.NUM_OBSTACLES)
            return gym.spaces.Box(low=-np.inf * np.ones(OBS_SIZE),
                                  high=np.inf * np.ones(OBS_SIZE),
                                  dtype=np.float32)
        else:
            raise ValueError('')

    def _compute_obs(self):
        """Compute the environment observation's from its internal state.

        Returns
        -------
        ndarray
            The observation of the environment at the current step.

        """
        if self.OBS_TYPE == ObsType.GLOBAL:
            temp_obs = []
            for i in range(self.NUM_TT):
                temp_obs = np.hstack([temp_obs, self.state['tt_'+str(i)]['id'], self.state['tt_'+str(i)]['class'], self.state['tt_'+str(i)]['pos']])
            for i in range(self.NUM_S1):
                temp_obs = np.hstack([temp_obs, self.state['s1_'+str(i)]['id'], self.state['s1_'+str(i)]['class'], self.state['s1_'+str(i)]['pos']])
            for i in range(self.NUM_ADVERSARY):
                # Only add visible adversaries.
                if  self.state['adv_'+str(i)]['visibility']:
                    temp_obs = np.hstack([temp_obs, self.state['adv_'+str(i)]['id'], self.state['adv_'+str(i)]['class'], self.state['adv_'+str(i)]['pos']])
                else:
                    temp_obs = np.hstack([temp_obs, 0, 0, [0, 0, 0]])
            for i in range(self.NUM_NEUTRAL):
                # Only add visible neutral agents.
                if  self.state['neu_'+str(i)]['visibility']:
                    temp_obs = np.hstack([temp_obs, self.state['neu_'+str(i)]['id'], self.state['neu_'+str(i)]['class'], self.state['neu_'+str(i)]['pos']])
                else:
                    temp_obs = np.hstack([temp_obs, 0, 0, [0, 0, 0]])
            for i in range(self.NUM_OBSTACLES):
                temp_obs = np.hstack([temp_obs, self.state['obst_'+str(i)]['id'], self.state['obst_'+str(i)]['class'], self.state['obst_'+str(i)]['pos']]) 
            return temp_obs
        else:
            raise ValueError('')

    def _compute_reward(self):
        """Compute the environment's reward signal.

        Returns
        -------
        float
            The reward signal of the environment at the current step.

        """
        if self.REWARD_CHOICE == RewardChoice.REWARD_A:
            reward = 0
            # Get the tellos and robomasters positions.
            tt_s1_positions = np.zeros((self.NUM_TT + self.NUM_S1, 3))
            for i in range(self.NUM_TT):
                pos, _ = p.getBasePositionAndOrientation(self.TT_IDS[i], physicsClientId=self.PYB_CLIENT)
                tt_s1_positions[i, :] = pos
            for i in range(self.NUM_S1):
                pos, _ = p.getBasePositionAndOrientation(self.S1_IDS[i], physicsClientId=self.PYB_CLIENT)
                tt_s1_positions[self.NUM_TT + i, :] = pos
            # For every adversary check whether there is a nearby tello or robomaster.
            for i in range(self.NUM_ADVERSARY):
                pos1, _ = p.getBasePositionAndOrientation(self.ADVERSARY_IDS[i], physicsClientId=self.PYB_CLIENT)
                dist = np.linalg.norm(tt_s1_positions - np.array(pos1), axis=1)
                if min(dist) < 1.5:  # Within the tellos' altitude.
                    reward += 1
            return reward
        else:
            raise ValueError('')

    def _apply_action(self,
                      action
                      ):
        """Set (in PyBullet) the velocities of the learning agents based on the input action.

        Parameters
        ----------
        action : ndarray
            The action selected by a controller, learning agent.

        """
        if self.ACTION_TYPE == ActionType.TASK_ASSIGNMENT:
            for i, val in enumerate(action):
                # NOTE: this relies on order the assets are created in .reset()
                agent_id = i + 1 + self.NUM_OBSTACLES
                target_id = val + 1 + self.NUM_OBSTACLES + self.NUM_TT + self.NUM_S1
                pos1, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=self.PYB_CLIENT)
                pos2, _ = p.getBasePositionAndOrientation(target_id, physicsClientId=self.PYB_CLIENT)
                connecting_vector = np.array(pos2[0:2]) - np.array(pos1[0:2])
                # Move along the connecting direction.
                vel = 2 * (connecting_vector / np.linalg.norm(connecting_vector))
                if i < self.NUM_TT:  # Tellos.
                    p.resetBaseVelocity(objectUniqueId=agent_id,
                                        linearVelocity=[vel[0], vel[1], 0],
                                        angularVelocity=[0, 0, 0],
                                        physicsClientId=self.PYB_CLIENT
                                        )
                else:  # Robomasters.
                    fall, _ = p.getBaseVelocity(agent_id, physicsClientId=self.PYB_CLIENT)
                    vel *= 1.5  # Speed correction.
                    p.resetBaseVelocity(objectUniqueId=agent_id,
                                        linearVelocity=[vel[0], vel[1], fall[2]],
                                        angularVelocity=[0, 0, 0],
                                        physicsClientId=self.PYB_CLIENT
                                        )
                if self.DEBUG:
                    p.addUserDebugLine(lineFromXYZ=pos1,
                                       lineToXYZ=pos2,
                                       lineColorRGB=[1, 0, 0],
                                       lifeTime=2 * self.CTRL_TIMESTEP,
                                       physicsClientId=self.PYB_CLIENT)
        elif self.ACTION_TYPE == ActionType.TRACKING:
            for i, val in enumerate(action):
                # NOTE: this relies on order the assets are created in .reset()
                agent_id = i + 1 + self.NUM_OBSTACLES
                if val == 0:
                    vel = [0, 0]  # Stay.
                elif val == 1:
                    vel = [1, 1]  # Forward.
                elif val == 2:
                    vel = [-1, -1]  # Backward.
                elif val == 3:
                    vel = [-1, 1]  # Left.
                elif val == 4:
                    vel = [1, -1]  # Right.
                else:
                    raise ValueError('')
                vel = 1.2 * np.array(vel)  # Speed correction.
                if i < self.NUM_TT:  # Tellos.
                    p.resetBaseVelocity(objectUniqueId=agent_id,
                                        linearVelocity=[vel[0], vel[1], 0],
                                        angularVelocity=[0, 0, 0],
                                        physicsClientId=self.PYB_CLIENT
                                        )
                else:  # Robomasters.
                    fall, _ = p.getBaseVelocity(agent_id, physicsClientId=self.PYB_CLIENT)
                    p.resetBaseVelocity(objectUniqueId=agent_id,
                                        linearVelocity=[vel[0], vel[1], fall[2]],
                                        angularVelocity=[0, 0, 0],
                                        physicsClientId=self.PYB_CLIENT
                                        )
        else:
            raise ValueError('')

    def _nonstationary_adversary(self):
        """Set (in PyBullet) the velocities of the non-learning adversarial agents.

        """
        # Move on a specified path.
        if self.ADV_TYPE == AdversaryType.BLIND:
            for i in range(self.NUM_ADVERSARY):
                if np.random.binomial(1, 0.05):
                    self.adversary_directions[i] = -self.adversary_directions[i]
                vel = self.adversary_directions[i] * self.ADVERSARY_SPEEDS[i] * np.array([1, -1])
                fall, _ = p.getBaseVelocity(self.ADVERSARY_IDS[i], physicsClientId=self.PYB_CLIENT)
                p.resetBaseVelocity(objectUniqueId=self.ADVERSARY_IDS[i],
                                    linearVelocity=[vel[0], vel[1], fall[2]],
                                    angularVelocity=[0, 0, 0],
                                    physicsClientId=self.PYB_CLIENT
                                    )
        # Move on a specified path but also flee an approaching tello or robomaster.
        elif self.ADV_TYPE == AdversaryType.AVOIDANT:
            tt_s1_positions = np.zeros((self.NUM_TT + self.NUM_S1, 3))
            for i in range(self.NUM_TT):
                pos, _ = p.getBasePositionAndOrientation(self.TT_IDS[i], physicsClientId=self.PYB_CLIENT)
                tt_s1_positions[i, :] = pos
            for i in range(self.NUM_S1):
                pos, _ = p.getBasePositionAndOrientation(self.S1_IDS[i], physicsClientId=self.PYB_CLIENT)
                tt_s1_positions[self.NUM_TT + i, :] = pos
            for i in range(self.NUM_ADVERSARY):
                pos1, _ = p.getBasePositionAndOrientation(self.ADVERSARY_IDS[i], physicsClientId=self.PYB_CLIENT)
                dist = np.linalg.norm(tt_s1_positions - np.array(pos1), axis=1)
                if np.any(np.squeeze(dist < 3)):  # Nearby TT/S1, closer than TT altitude.
                    closest_tt_s1_id = np.argmin(dist) + 1 + self.NUM_OBSTACLES
                    pos2, _ = p.getBasePositionAndOrientation(closest_tt_s1_id, physicsClientId=self.PYB_CLIENT)
                    connecting_vector = np.array(pos2[0:2]) - np.array(pos1[0:2])
                    # The escape velocity affect the effectiveness of a chase.
                    vel = -2.25 * self.ADVERSARY_SPEEDS[i] * (connecting_vector / np.linalg.norm(connecting_vector))
                else:
                    if np.random.binomial(1, 0.05):
                        self.adversary_directions[i] = -self.adversary_directions[i]
                    vel = self.adversary_directions[i] * self.ADVERSARY_SPEEDS[i] * np.array([1, -1])
                fall, _ = p.getBaseVelocity(self.ADVERSARY_IDS[i], physicsClientId=self.PYB_CLIENT)
                p.resetBaseVelocity(objectUniqueId=self.ADVERSARY_IDS[i],
                                    linearVelocity=[vel[0], vel[1], fall[2]],
                                    angularVelocity=[0, 0, 0],
                                    physicsClientId=self.PYB_CLIENT
                                    )
        else:
            raise ValueError('')

    def _nonstationary_neutral(self):
        """Set (in PyBullet) the velocities of the non-learning neutral agents.

        """
        # Move on a specified path.
        for i in range(self.NUM_NEUTRAL):
            if np.random.binomial(1, 0.05):
                self.neutral_directions[i] = -self.neutral_directions[i]
            vel = self.neutral_directions[i] * self.NEUTRAL_SPEEDS[i] * np.array([1, -1])
            fall, _ = p.getBaseVelocity(self.NEUTRAL_IDS[i], physicsClientId=self.PYB_CLIENT)
            p.resetBaseVelocity(objectUniqueId=self.NEUTRAL_IDS[i],
                                linearVelocity=[vel[0], vel[1], fall[2]],
                                angularVelocity=[0, 0, 0],
                                physicsClientId=self.PYB_CLIENT
                                )

    def _compute_done(self):
        """Compute the environment's termination flag.

        Returns
        -------
        bool
            Whether the episode ends at the current step.

        """
        # Check whether the episode has reached its maximum length.
        if (self.ctrl_step_counter + 1) / self.CTRL_FREQ >= self.EPISODE_LENGTH_SEC:
            return True
        else:
            return False

    def _update_env_state(self):
        """Update the internal state of the environment from PyBullet's data.

        """
        self.state = {}
        tt_s1_positions = np.zeros((self.NUM_TT + self.NUM_S1, 3))
        tt_s1_from_rays = []
        # Update the state of tellos and robomaster.
        for i in range(self.NUM_TT):
            temp_dict = {}
            temp_dict['id'] = self.TT_IDS[i]
            temp_dict['class'] = 1
            temp_dict['pos'], temp_dict['quat'] = p.getBasePositionAndOrientation(self.TT_IDS[i], physicsClientId=self.PYB_CLIENT)
            temp_dict['rpy'] = p.getEulerFromQuaternion(temp_dict['quat'])
            temp_dict['vel'], _ = p.getBaseVelocity(self.TT_IDS[i], physicsClientId=self.PYB_CLIENT)
            self.state['tt_'+str(i)] = temp_dict
            tt_s1_positions[i, :] = temp_dict['pos']
            tt_s1_from_rays += [temp_dict['pos']]
        for i in range(self.NUM_S1):
            temp_dict = {}
            temp_dict['id'] = self.S1_IDS[i]
            temp_dict['class'] = 2
            temp_dict['pos'], temp_dict['quat'] = p.getBasePositionAndOrientation(self.S1_IDS[i], physicsClientId=self.PYB_CLIENT)
            temp_dict['rpy'] = p.getEulerFromQuaternion(temp_dict['quat'])
            temp_dict['vel'], _ = p.getBaseVelocity(self.S1_IDS[i], physicsClientId=self.PYB_CLIENT)
            self.state['s1_'+str(i)] = temp_dict
            tt_s1_positions[self.NUM_TT + i, :] = temp_dict['pos']
            tt_s1_from_rays += [np.array(temp_dict['pos']) + np.array([0, 0, 0.35])]  # Start rays on top of the turret to avoid self-collisions.
        # Update the state of the adversaries.
        for i in range(self.NUM_ADVERSARY):
            temp_dict = {}
            temp_dict['id'] = self.ADVERSARY_IDS[i]
            temp_dict['class'] = 3
            temp_dict['pos'], temp_dict['quat'] = p.getBasePositionAndOrientation(self.ADVERSARY_IDS[i], physicsClientId=self.PYB_CLIENT)
            temp_dict['rpy'] = p.getEulerFromQuaternion(temp_dict['quat'])
            temp_dict['vel'], _ = p.getBaseVelocity(self.ADVERSARY_IDS[i], physicsClientId=self.PYB_CLIENT)
            # Check whether within the visibility range.
            dist = np.linalg.norm(tt_s1_positions - np.array(temp_dict['pos']), axis=1)
            in_range = np.squeeze(dist < self.VISIBILITY_THRESHOLD)
            # Check whether in line of sight.
            temp_repeated_to_rays = []
            for _ in range(tt_s1_positions.shape[0]):
                temp_repeated_to_rays += [temp_dict['pos']]
            if self.NUM_OBSTACLES == 0:
                in_sight = np.repeat(True, self.NUM_TT + self.NUM_S1)
                temp_dict['visibility'] = np.any(in_range)
            elif self.NUM_OBSTACLES > 0:
                rays = np.array(p.rayTestBatch(rayFromPositions=tt_s1_from_rays,
                                               rayToPositions=temp_repeated_to_rays,
                                               physicsClientId=self.PYB_CLIENT
                                               ))
                target_id = self.ADVERSARY_IDS[i]
                num_rays = tt_s1_positions.shape[0]
                in_sight = []
                for j in range(num_rays):
                    origin_id = self.TT_IDS[j] if (j < self.NUM_TT) else self.S1_IDS[j - self.NUM_TT]
                    if (rays[j, 0] == target_id):  # Consider the target 'in sight' if the ray strikes it.
                        in_sight += [True]
                    else:
                        in_sight += [False]
                in_sight = np.squeeze(in_sight)
                temp_dict['visibility'] = np.any(in_sight & in_range)
            else:
                raise ValueError('')
            self.state['adv_'+str(i)] = temp_dict
            # Add line-of-sight debug information on the GUI.
            if self.DEBUG:
                for j in range(tt_s1_positions.shape[0]):
                    if in_sight[j] and in_range[j]:
                        p.addUserDebugLine(lineFromXYZ=tt_s1_from_rays[j],
                                           lineToXYZ=temp_repeated_to_rays[j],
                                           lineColorRGB=[0, 1, 0],
                                           lifeTime=3 * self.CTRL_TIMESTEP,
                                           physicsClientId=self.PYB_CLIENT)
                    elif in_sight[j]:
                        p.addUserDebugLine(lineFromXYZ=tt_s1_from_rays[j],
                                           lineToXYZ=temp_repeated_to_rays[j],
                                           lineColorRGB=[0, 0, 1],
                                           lifeTime=3 * self.CTRL_TIMESTEP,
                                           physicsClientId=self.PYB_CLIENT)
                    else:
                        p.addUserDebugLine(lineFromXYZ=tt_s1_from_rays[j],
                                           lineToXYZ=temp_repeated_to_rays[j],
                                           lineColorRGB=[0.6, 0.6, 0.6],
                                           lifeTime=3 * self.CTRL_TIMESTEP,
                                           physicsClientId=self.PYB_CLIENT)
        # Update the state of the neutral agents.
        for i in range(self.NUM_NEUTRAL):            
            temp_dict = {}
            temp_dict['id'] = self.NEUTRAL_IDS[i]
            temp_dict['class'] = 4
            temp_dict['pos'], temp_dict['quat'] = p.getBasePositionAndOrientation(self.NEUTRAL_IDS[i], physicsClientId=self.PYB_CLIENT)
            temp_dict['rpy'] = p.getEulerFromQuaternion(temp_dict['quat'])
            temp_dict['vel'], _ = p.getBaseVelocity(self.NEUTRAL_IDS[i], physicsClientId=self.PYB_CLIENT)
            # Check whether within the visibility range.
            dist = np.linalg.norm(tt_s1_positions - np.array(temp_dict['pos']), axis=1)
            in_range = np.squeeze(dist < self.VISIBILITY_THRESHOLD)
            # Check whether in line of sight.
            temp_repeated_to_rays = []
            for _ in range(tt_s1_positions.shape[0]):
                temp_repeated_to_rays += [temp_dict['pos']]
            if self.NUM_OBSTACLES == 0:
                in_sight = np.repeat(True, self.NUM_TT + self.NUM_S1)
                temp_dict['visibility'] = np.any(in_range)
            elif self.NUM_OBSTACLES > 0:
                rays = np.array(p.rayTestBatch(rayFromPositions=tt_s1_from_rays,
                                               rayToPositions=temp_repeated_to_rays,
                                               physicsClientId=self.PYB_CLIENT
                                               ))
                target_id = self.NEUTRAL_IDS[i]
                num_rays = tt_s1_positions.shape[0]
                in_sight = []
                for j in range(num_rays):
                    origin_id = self.TT_IDS[j] if (j < self.NUM_TT) else self.S1_IDS[j - self.NUM_TT]
                    if (rays[j, 0] == target_id):  # Consider the target 'in sight' if the ray strikes it.
                        in_sight += [True]
                    else:
                        in_sight += [False]
                in_sight = np.squeeze(in_sight)
                temp_dict['visibility'] = np.any(in_sight & in_range)
            else:
                raise ValueError('')
            self.state['neu_'+str(i)] = temp_dict
            # Add line-of-sight debug information on the GUI.
            if self.DEBUG:
                for j in range(tt_s1_positions.shape[0]):
                    if in_sight[j] and in_range[j]:
                        p.addUserDebugLine(lineFromXYZ=tt_s1_from_rays[j],
                                           lineToXYZ=temp_repeated_to_rays[j],
                                           lineColorRGB=[0, 1, 0],
                                           lifeTime=3 * self.CTRL_TIMESTEP,
                                           physicsClientId=self.PYB_CLIENT)
                    elif in_sight[j]:
                        p.addUserDebugLine(lineFromXYZ=tt_s1_from_rays[j],
                                           lineToXYZ=temp_repeated_to_rays[j],
                                           lineColorRGB=[0, 0, 1],
                                           lifeTime=3 * self.CTRL_TIMESTEP,
                                           physicsClientId=self.PYB_CLIENT)
                    else:
                        p.addUserDebugLine(lineFromXYZ=tt_s1_from_rays[j],
                                           lineToXYZ=temp_repeated_to_rays[j],
                                           lineColorRGB=[0.6, 0.6, 0.6],
                                           lifeTime=3 * self.CTRL_TIMESTEP,
                                           physicsClientId=self.PYB_CLIENT)
        # Include the obstacles (that can be moved) to the state.
        for i in range(self.NUM_OBSTACLES):
            temp_dict = {}
            temp_dict['id'] = self.OBSTACLES_IDS[i]
            temp_dict['class'] = 5
            temp_dict['size'] = 1
            temp_dict['pos'], _, _, _, _, _ = p.getLinkState(self.OBSTACLES_IDS[i], linkIndex=1, physicsClientId=self.PYB_CLIENT)
            self.state['obst_'+str(i)] = temp_dict

    def _create_obstacles(self):
        """Load the obstacles' URDF files.

        Returns
        -------
        ndarray
            The PyBullet IDs of the obstacles.

        """
        return np.array([
            p.loadURDF(fileName='cube_no_rotation.urdf',
                       basePosition=[
                                     np.random.uniform(low=self.ARENA_EDGE/3, high=2*self.ARENA_EDGE/3),
                                     np.random.uniform(low=self.ARENA_EDGE/3, high=2*self.ARENA_EDGE/3),
                                     0
                                     ],
                       baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.PYB_CLIENT
                       )
            for i in range(self.NUM_OBSTACLES)
            ])

    def _create_team_tt(self):
        """Load the tellos' URDF files.

        Returns
        -------
        ndarray
            The PyBullet IDs of the tellos.

        """
        return np.array([
            p.loadURDF(fileName=os.path.dirname(os.path.abspath(__file__))+'/../urdf_and_meshes/'+'tello.urdf',
                       basePosition=[
                                     np.random.uniform(low=0, high=self.ARENA_EDGE/3),
                                     np.random.uniform(low=0, high=self.ARENA_EDGE/3),
                                     1.5
                                     ],
                       baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                       # flags=p.URDF_USE_INERTIA_FROM_FILE,
                       physicsClientId=self.PYB_CLIENT
                       )
            for i in range(self.NUM_TT)
            ])

    def _create_team_s1(self):
        """Load the robomasters' URDF files.

        Returns
        -------
        ndarray
            The PyBullet IDs of the robomasters.

        """
        return np.array([
            p.loadURDF(fileName=os.path.dirname(os.path.abspath(__file__))+'/../urdf_and_meshes/'+'robomaster.urdf',
                       basePosition=[
                                     np.random.uniform(low=0, high=self.ARENA_EDGE/3),
                                     np.random.uniform(low=0, high=self.ARENA_EDGE/3),
                                     0.5
                                     ],
                       baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                       # flags=p.URDF_USE_INERTIA_FROM_FILE,
                       physicsClientId=self.PYB_CLIENT
                       )
            for i in range(self.NUM_S1)
            ])

    def _create_adversary(self):
        """Load the adversaries' URDF files.

        Returns
        -------
        ndarray
            The PyBullet IDs of the adversaries.

        """
        return np.array([
            p.loadURDF(fileName=os.path.dirname(os.path.abspath(__file__))+'/../urdf_and_meshes/'+'adversary.urdf',
                       basePosition=[
                                     np.random.uniform(low=2*self.ARENA_EDGE/3, high=self.ARENA_EDGE),
                                     np.random.uniform(low=2*self.ARENA_EDGE/3, high=self.ARENA_EDGE),
                                     0.5
                                     ],
                       baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                       # flags=p.URDF_USE_INERTIA_FROM_FILE,
                       physicsClientId=self.PYB_CLIENT
                       )
            for i in range(self.NUM_ADVERSARY)
            ])

    def _create_neutral(self):
        """Load the neutral agents' URDF files.

        Returns
        -------
        ndarray
            The PyBullet IDs of the neutral agents.

        """
        return np.array([
            p.loadURDF(fileName=os.path.dirname(os.path.abspath(__file__))+'/../urdf_and_meshes/'+'neutral.urdf',
                       basePosition=[
                                     np.random.uniform(low=2*self.ARENA_EDGE/3, high=self.ARENA_EDGE),
                                     np.random.uniform(low=2*self.ARENA_EDGE/3, high=self.ARENA_EDGE),
                                     0.5
                                     ],
                       baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                       # flags=p.URDF_USE_INERTIA_FROM_FILE,
                       physicsClientId=self.PYB_CLIENT
                       )
            for i in range(self.NUM_NEUTRAL)
            ])

    def _debug_ids(self):
        """Add debug text (the robots' IDs) to PyBullet GUI's.

        """
        for i in range(self.NUM_OBSTACLES):
            pos, _, _, _, _, _ = p.getLinkState(self.OBSTACLES_IDS[i], linkIndex=1, physicsClientId=self.PYB_CLIENT)
            self.OBSTACLES_DEBUG_LABELS[i] = p.addUserDebugText(text=str(self.OBSTACLES_IDS[i]),
                                                                textPosition=np.array(pos)+np.array([0, 0, 1.2]),
                                                                textColorRGB=[0, 0, 0],
                                                                replaceItemUniqueId=int(self.OBSTACLES_DEBUG_LABELS[i]),
                                                                physicsClientId=self.PYB_CLIENT)
        for i in range(self.NUM_TT):
            pos, _ = p.getBasePositionAndOrientation(self.TT_IDS[i], physicsClientId=self.PYB_CLIENT)
            self.TT_DEBUG_LABELS[i] = p.addUserDebugText(text=str(self.TT_IDS[i]),
                                                         textPosition=np.array(pos)+np.array([0, 0, 0.2]),
                                                         textColorRGB=[0, 0, 1],
                                                         replaceItemUniqueId=int(self.TT_DEBUG_LABELS[i]),
                                                         physicsClientId=self.PYB_CLIENT)
        for i in range(self.NUM_S1):
            pos, _ = p.getBasePositionAndOrientation(self.S1_IDS[i], physicsClientId=self.PYB_CLIENT)
            self.S1_DEBUG_LABELS[i] = p.addUserDebugText(text=str(self.S1_IDS[i]),
                                                         textPosition=np.array(pos)+np.array([0, 0, 0.4]),
                                                         textColorRGB=[0, 0, 1],
                                                         replaceItemUniqueId=int(self.S1_DEBUG_LABELS[i]),
                                                         physicsClientId=self.PYB_CLIENT)
        for i in range(self.NUM_ADVERSARY):
            pos, _ = p.getBasePositionAndOrientation(self.ADVERSARY_IDS[i], physicsClientId=self.PYB_CLIENT)
            label = str(self.ADVERSARY_IDS[i]) if self.state['adv_'+str(i)]['visibility'] else '('+str(self.ADVERSARY_IDS[i])+')'
            self.ADVERSARY_DEBUG_LABELS[i] = p.addUserDebugText(text=label,
                                                                textPosition=np.array(pos)+np.array([0, 0, 0.4]),
                                                                textColorRGB=[1, 0, 0],
                                                                replaceItemUniqueId=int(self.ADVERSARY_DEBUG_LABELS[i]),
                                                                physicsClientId=self.PYB_CLIENT)
        for i in range(self.NUM_NEUTRAL):
            pos, _ = p.getBasePositionAndOrientation(self.NEUTRAL_IDS[i], physicsClientId=self.PYB_CLIENT)
            label = str(self.NEUTRAL_IDS[i]) if self.state['neu_'+str(i)]['visibility'] else '('+str(self.NEUTRAL_IDS[i])+')'
            self.NEUTRAL_DEBUG_LABELS[i] = p.addUserDebugText(text=label,
                                                              textPosition=np.array(pos)+np.array([0, 0, 0.4]),
                                                              textColorRGB=[0.5, 0.5, 0.5],
                                                              replaceItemUniqueId=int(self.NEUTRAL_DEBUG_LABELS[i]),
                                                              physicsClientId=self.PYB_CLIENT)
