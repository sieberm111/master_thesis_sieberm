# ! /usr/bin/env python
import wandb
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import random
import os
import time
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

import rospy
from gazebo_msgs.srv import GetLinkState, GetJointProperties, SetPhysicsProperties
from gazebo_msgs.srv import SetPhysicsPropertiesRequest, SpawnModel, DeleteModel
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from gazebo_msgs.msg import ODEPhysics
import torch
# import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!

# golie slide = 0 - 0.223
# def slide = 0 - 0.415
# att slide = 0 - 0.275
# rotate all = -3.14 - 3.14

# rotate + rotace utok
# rotate - rotace defend

# y_ball min:max -0.68:0 leva:prava
# x_ball min:max -0.05:0.45 obrana:utok


# SHOT dictionary
shot_dict = {'mid_shot': {'pos_rot': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
                          'pos_slide': [0., 0.05, 0.07, 0.08, 0.1, 0.12, 0.1475, 0.1475, 0.1475, 0.1475],
                          'ball_pos': Point(0.4, -0.35, 0.05)},
             'mid_left':{'pos_rot': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
                          'pos_slide': [0., 0.05, 0.07, 0.08, 0.1, 0.1, 0.1,0.1,0.1,0.1],
                          'ball_pos': Point(0.39, -0.40, 0.05)},
             'diagonal':{'pos_rot': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5,
                -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                          'pos_slide': [0., 0.003552632, 0.007105263, 0.010657895, 0.014210526,
             0.017763158, 0.021315789, 0.024868421, 0.028421053, 0.031973684,
             0.035526316, 0.039078947, 0.042631579, 0.046184211, 0.0474,
             0.053289474, 0.056842105, 0.060394737, 0.063947368, 0.0474, 0.0474, 0.0472, 0.0475, 0.0475, 0.0475],
                          'ball_pos': Point(0.39, -0.25, 0.05)},
             'miss':{'pos_rot': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
                          'pos_slide': [0., 0.05, 0.07, 0.08, 0.1, 0.1, 0.1,0.1,0.1,0.1],
                          'ball_pos': Point(0.39, -0.6, 0.05)},
             'miss_sneaky': {'pos_rot': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
                         'pos_slide': [0., 0.05, 0.07, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                          'ball_pos': Point(0.39, -0.6, 0.05)},
             'potah':{'pos_rot': [0,0,0,0,0, -1.5, -1.5, -1.5, -1.3, -1, 1, 1.5, 1.5],
                          'pos_slide': [0., 0.01,0.04,0.07,0.08,0.1, 0.11,0.15,0.11,0.11,0.1,0.1,0.1],
                          'ball_pos': Point(0.39, -0.6, 0.05)}}


class FutEnv(Env):
    def __init__(self):
        # Actions we can take discrete 0-99 for slide a rot of def and golie
        self.action_space = MultiDiscrete([100, 100, 100, 100])
        self.observation_space = Box(low=-1.1, high=1.1, shape=(4,), dtype=np.float32)
        # Set start state
        self.state = np.zeros(4)  # [pos_att; rot_att; x_ball; y_ball ]
        # Set game length
        self.game_length = 64
        # Set eval data for reward calc
        self.eval_data = np.zeros(11)  # all relative pos
        # Set last action
        self.last_action = [0, 0, 0, 0]
        #iterator for shot sequence
        self.shot_iterator = 0
        #cumulative reward for game
        self.cumul_reward = 0
        # set shot params
        self.random_shot = random.choice(list(shot_dict.keys()))
        self.pos_rot = shot_dict[self.random_shot]['pos_rot']
        self.pos_slide = shot_dict[self.random_shot]['pos_slide']
        self.ball_pos = shot_dict[self.random_shot]['ball_pos']
        # logging variables
        self.shot_counter = 0
        self.episodes_counter = 0

        # subscribe gazebo
        rospy.init_node('subscriber', anonymous=True)
        self.rate = rospy.Rate(rospy.get_param('~publish_rate', 10))
        # talk to gazebo
        try:
            self.pub_att_rev = rospy.Publisher('/futfullv5/rev_att_position_controller/command', Float64, queue_size=10)
            self.pub_att_slide = rospy.Publisher('/futfullv5/slide_att_position_controller/command', Float64,
                                                 queue_size=10)
            self.pub_def_rev = rospy.Publisher('/futfullv5/rev_def_position_controller/command', Float64, queue_size=10)
            self.pub_def_slide = rospy.Publisher('/futfullv5/slide_def_position_controller/command', Float64,
                                                 queue_size=10)
            self.pub_goal_rev = rospy.Publisher('/futfullv5/rev_golie_position_controller/command', Float64,
                                                queue_size=10)
            self.pub_goal_slide = rospy.Publisher('/futfullv5/slide_golie_position_controller/command', Float64,
                                                  queue_size=10)
        except rospy.ServiceException as e:
            rospy.loginfo("Talker init call failed:  {0}".format(e))

        # Pause and unpause physics
        rospy.wait_for_service('/gazebo/pause_physics')
        rospy.wait_for_service('/gazebo/unpause_physics')

        self.pause_physics_client = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics_client = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    def transform_range(self, list_coord):
        list_out = np.zeros(4)
        list_out[0] = (list_coord[2] * 7.27272727272727) - 1  # mapuje rozsah na -1:1 att slide
        list_out[1] = list_coord[5] / 1.57  # mapuje rozsah na -1:1 att rotation
        list_out[2] = (list_coord[6] * 4) - 0.9  # maps x coordinate of ball to -1:1 (1 close to att, -1 close to goal)
        list_out[3] = (list_coord[7] * 2.92825768667643) + 1  # maps y of ball to -1:1 left:right
        return list_out

    def transform_coordinates(self, list_coord):
        # [slide_golie, slide_def, slide_att, rev_golie, rev_def, rev_att, ball_x, ball_y]
        list_out = np.zeros(11)
        list_out[0] = ((list_coord[0] - 0.4691) * 2.92825768667643) + 1  # mapuje golie slide na pos v hristi -1:1
        list_out[1] = ((list_coord[1] - 0.6811) * 2.92825768667643) + 1  # mapuje def slide leva na pos v hristi -1:1
        list_out[2] = ((list_coord[1] - 0.451) * 2.92825768667643) + 1  # mapuje def slide prava na pos v hristi -1:1
        list_out[3] = ((list_coord[2] - 0.6811) * 2.92825768667643) + 1  # mapuje rozsah na 0:0.28 att slide leva na pos v hristi pak na -1:1 v y
        list_out[4] = ((list_coord[2] - 0.4961) * 2.92825768667643) + 1  # mapuje rozsah na 0:0.28 att slide stred na pos v hristi pak na -1:1 v y
        list_out[5] = ((list_coord[2] - 0.3111) * 2.92825768667643) + 1  # mapuje rozsah na 0:0.28 att slide prava na pos v hristi pak na -1:1 v y

        list_out[6] = list_coord[3] / 1.57  # mapuje rozsah na -1:1 golie rotation
        list_out[7] = list_coord[4] / 1.57  # mapuje rozsah na -1:1 def rotation
        list_out[8] = list_coord[5] / 1.57  # mapuje rozsah na -1:1 att rotation

        list_out[9] = (list_coord[6] * 4) - 0.8  # maps x coordinate of ball to -1:1 (1 close to att, -1 close to goal)
        list_out[10] = (list_coord[7] * 2.92825768667643) + 1  # maps y of ball to -1:1 left:right
        #print(list_out)
        return list_out

    def show_gazebo_models(self):
        try:
            ball_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
            ball_coordinates_out = ball_coordinates('ball::ball', '')
            # print('\n')
            # print('Status.success = ', ball_coordinates_out.success)
            # print("ball X : " + str(ball_coordinates_out.link_state.pose.position.x))
            # print("ball Y : " + str(ball_coordinates_out.link_state.pose.position.y))

            #respawn ball model if failed
            if ball_coordinates_out.success == False:
                wandb.log({"ball_respawn": 1})
                rospy.wait_for_service('/gazebo/spawn_urdf_model')
                spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
                try:
                    spawn_model(
                        model_name='ball',
                        model_xml=open('/home/toofy/catkin_ws/src/futfullv5_description/urdf/ball.xacro', 'r').read(),
                        robot_namespace='',
                        initial_pose=Pose(position=self.ball_pos, orientation=Quaternion(0, 0, 0, 0)),
                        reference_frame='world')

                except rospy.ServiceException as e:
                    print("Service call failed: ", e)

            slide_golie = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
            slide_golie_out = slide_golie('slide_golie')

            slide_att = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
            slide_att_out = slide_att('slide_att')

            slide_def = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
            slide_def_out = slide_def('slide_def')

            rev_golie = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
            rev_golie_out = rev_golie('rev_golie')

            rev_att = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
            rev_att_out = rev_att('rev_att')

            rev_def = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
            rev_def_out = rev_def('rev_def')

            # print("golie- slide: " + str(slide_golie_out.position[0]) + " rev: " + str(rev_golie_out.position[0]))
            # print("att- slide: " + str(slide_att_out.position[0]) + " rev: "+ str(rev_att_out.position[0]))
            # print("def- slide: " + str(slide_def_out.position[0]) + " rev: " + str(rev_def_out.position[0]))

            coord_list_t = [slide_golie_out.position[0], slide_def_out.position[0], slide_att_out.position[0],
                          rev_golie_out.position[0], rev_def_out.position[0], rev_att_out.position[0],
                          ball_coordinates_out.link_state.pose.position.x,
                          ball_coordinates_out.link_state.pose.position.y]
        except rospy.ServiceException as e:
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))
        return [np.array(self.transform_coordinates(coord_list_t)).astype(np.float32),
                np.array(self.transform_range(coord_list_t)).astype(np.float32)]

    def publish_data(self, pos_att_rev, pos_att_slide, pos_def_rev, pos_def_slide, pos_golie_rev, pos_golie_slide):
        try:
            self.pub_att_rev.publish(pos_att_rev)
            self.pub_att_slide.publish(pos_att_slide)
            self.pub_def_rev.publish(pos_def_rev)
            self.pub_def_slide.publish(pos_def_slide)
            self.pub_goal_rev.publish(pos_golie_rev)
            self.pub_goal_slide.publish(pos_golie_slide)

        except rospy.ServiceException as e:
            rospy.loginfo("Publish data call failed:  {0}".format(e))

    def transform_coordinates_out(self, list_coord):
        list_out = np.zeros(4)
        list_out[0] = list_coord[0] * 0.0022222222222222222  # mapuje output z 0:99  na 0:0.22 golie slide
        list_out[1] = list_coord[1] * 0.004141414141414141  # mapuje rozsah na 0:99 def slide

        list_out[2] = (list_coord[2] * 0.03171717171717172) - 1.57  # mapuje rozsah na -1.57:1.57 golie rotation
        list_out[3] = (list_coord[3] * 0.03171717171717172) - 1.57  # mapuje rozsah na -1.57:1.57 def rotation
        return list_out

    def step(self, action):
        # Transform data out
        coord_list_out = env.transform_coordinates_out(action)

        # Publish data
        if self.shot_iterator < len(self.pos_rot):
            env.publish_data(self.pos_rot[self.shot_iterator], self.pos_slide[self.shot_iterator],
                             coord_list_out[3], coord_list_out[1], coord_list_out[2], coord_list_out[0])
        else:
            env.publish_data(0, random.randint(0, 20) / 100, coord_list_out[3], coord_list_out[1], coord_list_out[2],
                             coord_list_out[0])
            #env.publish_data(0, random.randint(0, 20) / 100, 0, coord_list_out[1], 0, coord_list_out[0])
        # Wait for period
        env.rate.sleep()
        # Get positions of players and ball
        [self.eval_data, self.state] = self.show_gazebo_models()
        # Reduce game length by 1 second
        self.game_length -= 1

        # Calculate reward
        reward = 0
        # Epsilon for overlap of players
        eps = 0.009
        # Penalty for moving
        action_penalty = abs(self.last_action[0] - action[0]) + abs(self.last_action[1] - action[1]) + \
                         abs(self.last_action[2] - action[2]) + abs(self.last_action[3] - action[3]) #penalty rotation

        if self.eval_data[10] < 0.3:
            reward -= action_penalty/3
        else:
            reward -= action_penalty
        # Reward if players are in front of ball
        if self.eval_data[0] + eps <= self.eval_data[10] <= self.eval_data[0] - eps:
            reward += 10
        if self.eval_data[1] + eps <= self.eval_data[10] <= self.eval_data[1] - eps:
            reward += 10
        if self.eval_data[2] + eps <= self.eval_data[10] <= self.eval_data[2] - eps:
            reward += 10

        if 0.2 < self.eval_data[6] < -0.2: #penalty velka rotace
            reward -= 100
        if 0.5 < self.eval_data[7] < -0.5:
            reward -= 50

        # Check if game is done
        if self.eval_data[9] > 1:  # ball pos za utocnikama
            wandb.log({self.random_shot: 1})
            wandb.log({"x_axis_shot": self.shot_counter})
            self.shot_counter += 1
            done = True
            reward += 50
        elif self.eval_data[9] < -1.05:  # ball pos gol
            wandb.log({self.random_shot: -1})
            wandb.log({"x_axis_shot": self.shot_counter})
            self.shot_counter += 1
            done = True
            reward -= 500
        elif self.game_length == 0:
            wandb.log({self.random_shot: 0})
            wandb.log({"x_axis_shot": self.shot_counter})
            self.shot_counter += 1
            done = True
        else:
            done = False

        self.shot_iterator += 1
        self.last_action = action
        # Set placeholder for info
        info = {}
        wandb.log({"reward": reward})
        self.cumul_reward += reward
        # Return step information

        # print("rev: {}", reward)
        # print("ball_pos: {}", self.eval_data[10])
        return self.state, float(reward), done, info

    def render(self):
        # Implement viz
        pass

    def reset(self):
        # reset gazebo
        self.publish_data(0, 0, 0, 0, 0, 0)
        rospy.wait_for_service('/gazebo/reset_world')
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_world()
        # Reset state
        self.state = np.array([0, 0, 0, 0]).astype(dtype=np.float32)
        # Reset game time
        self.game_length = 64
        self.shot_iterator = 0

        # random shot select
        self.random_shot = random.choice(list(shot_dict.keys()))
        self.pos_rot = shot_dict[self.random_shot]['pos_rot']
        self.pos_slide = shot_dict[self.random_shot]['pos_slide']
        self.ball_pos = shot_dict[self.random_shot]['ball_pos']

        # delete ball
        rospy.wait_for_service('/gazebo/delete_model')
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        try:
            delete_model(model_name="ball")
        except rospy.ServiceException as e:
            print("Service call failed: ", e)
        rospy.wait_for_service('/gazebo/delete_model')
        time.sleep(0.05)
        wandb.log({"game_reward": self.cumul_reward})
        self.cumul_reward = 0

        #spawn model
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        try:
            spawn_model(
                model_name='ball',
                model_xml=open('/home/toofy/catkin_ws/src/futfullv5_description/urdf/ball.xacro', 'r').read(),
                robot_namespace='',
                initial_pose=Pose(position=self.ball_pos, orientation=Quaternion(0, 0, 0, 0)),
                reference_frame='world')

        except rospy.ServiceException as e:
            print("Service call failed: ", e)
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        time.sleep(0.05)
        wandb.log({"episodes": self.episodes_counter})
        self.episodes_counter += 1

        return self.state

def setup_gazebo_physics():
    rospy.wait_for_service('/gazebo/set_physics_properties')
    set_physics_properties = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)

    gravity = Vector3()
    gravity.x = 0.0
    gravity.y = 0.0
    gravity.z = -9.800

    ode_config = ODEPhysics()
    ode_config.auto_disable_bodies = False
    ode_config.sor_pgs_precon_iters = 0
    ode_config.sor_pgs_iters = 50
    ode_config.sor_pgs_w = 1.3
    ode_config.sor_pgs_rms_error_tol = 0.0
    ode_config.contact_surface_layer = 0.001
    ode_config.contact_max_correcting_vel = 0.0
    ode_config.cfm = 0.0
    ode_config.erp = 0.2
    ode_config.max_contacts = 20

    set_physics_request = SetPhysicsPropertiesRequest()
    set_physics_request.time_step = float(0.001)
    set_physics_request.max_update_rate = float(0.0)
    set_physics_request.gravity = gravity
    set_physics_request.ode_config = ode_config

    res = set_physics_properties(set_physics_request)
    print(res)


if __name__ == '__main__':
    #wanb init

    wandb.init(project="master_thesis", entity="sieberm", name="PPO_500K_multi_shot_rot_overtrain1")
    wandb.config = {
        "learning_rate": 0.003,
        "time_steps": 500000,
        "batch_size": 64
    }
    wandb.run.define_metric("game_reward", step_metric="episodes", goal="maximize")
    # ROS physics properties set
    setup_gazebo_physics()

    #init Enviroment
    env = FutEnv()
    #check_env(env, warn=True)
    env.unpause_physics_client()

    #init internal loging and saving
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./saved_models_multishot_rot/',
                                             name_prefix='rl_model_over')

    #model init
    #model = PPO("MlpPolicy", env, verbose=1)
    model = PPO.load("./saved_models_multishot_rot/PPO_NoRotation_multishot_rot", env, verbose=1)
    #model(env=env,verbose=1)
    # Train the agent
    model.learn(total_timesteps=500_000, callback=checkpoint_callback)
    # Save the agent
    model.save("saved_models_multishot_rot/PPO_NoRotation_multishot_rot_over")

    env.pause_physics_client()