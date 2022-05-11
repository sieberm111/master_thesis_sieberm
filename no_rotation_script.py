# ! /usr/bin/env python
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import random
import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.srv import GetJointProperties
import rospy
from std_msgs.msg import Float64
from std_srvs.srv import Empty
import torch
# import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
import numpy as np

# golie slide = 0 - 0.22
# def slide = 0 - 0.41 mozna 0.42
# att slide = 0 - 0.28
# rotate all = -3.14 - 3.14
# rotate + rotace utok
# rotate - rotace defend
# y_ball min:max -0.68:0 leva:prava
# x_ball min:max -0.05:0.45 obrana:utok
# pos_rot = [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5,
#            -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
# pos_slide = [0., 0.003552632, 0.007105263, 0.010657895, 0.014210526,
#              0.017763158, 0.021315789, 0.024868421, 0.028421053, 0.031973684,
#              0.035526316, 0.039078947, 0.042631579, 0.046184211, 0.0474,
#              0.053289474, 0.056842105, 0.060394737, 0.063947368, 0.0474, 0.0474, 0.0472, 0.0475, 0.0475, 0.0475]
pos_rot =   [-1.5, -1.5, -1.5, -1.5, -1.5, -0.95, 1.5, 1.5, 1.5] #straight shot middle
pos_slide = [0., 0., 0.5, 0.12, 0.13, 0.14, 0.15, 0.15, 0.15, 0.15]

class FutEnv(Env):
    def __init__(self):
        # Actions we can take discrete 0-99 for slide a rot of def and golie
        self.action_space = MultiDiscrete([100, 100, 100, 100])
        self.observation_space = Box(low=-1.1, high=1.1, shape=(4,), dtype=np.float32)
        # Set start state
        self.state = np.zeros(4)  # [pos_att; rot_att; x_ball; y_ball ]
        # Set game length
        self.game_length = 60
        # Set eval data for reward calc
        self.eval_data = np.zeros(11)  # all relative pos
        # Set last action
        self.last_action = [0, 0, 0, 0]
        #iterator for shot sequence
        self.shot_iterator = 0
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

    def transform_range(self, list_coord):
        list_out = np.zeros(4)
        list_out[0] = ((list_coord[2] * 3.571428571) * 2) - 1  # mapuje rozsah na -1:1 att slide
        list_out[1] = list_coord[5] / 1.57  # mapuje rozsah na -1:1 att rotation
        list_out[2] = (list_coord[6] * 4) - 0.8  # maps x coordinate of ball to -1:1 (1 close to att, -1 close to goal)
        list_out[3] = (list_coord[7] * 2.93255131964809) + 1  # maps y of ball to -1:1 left:right
        return list_out

    def transform_coordinates(self, list_coord):
        list_out = np.zeros(11)
        list_out[0] = ((list_coord[
                            0] - 0.47) * 2.93255131964809) + 1  # mapuje rozsah na 0:0.22 golie slide na pos v hristi pak na -1:1 v y
        list_out[1] = ((list_coord[
                            1] - 0.682) * 2.93255131964809) + 1  # mapuje rozsah na 0:0.41 def slide leva na pos v hristi pak na -1:1 v y
        list_out[2] = ((list_coord[
                            1] - 0.265) * 2.93255131964809) + 1  # mapuje rozsah na 0:0.41 def slide prava na pos v hristi pak na -1:1 v y
        list_out[3] = ((list_coord[
                            2] - 0.682) * 2.93255131964809) + 1  # mapuje rozsah na 0:0.28 att slide leva na pos v hristi pak na -1:1 v y
        list_out[4] = ((list_coord[
                            2] - 0.312) * 2.93255131964809) + 1  # mapuje rozsah na 0:0.28 att slide stred na pos v hristi pak na -1:1 v y
        list_out[5] = ((list_coord[
                            2] - 0.497) * 2.93255131964809) + 1  # mapuje rozsah na 0:0.28 att slide prava na pos v hristi pak na -1:1 v y

        list_out[6] = list_coord[3] / 1.57  # mapuje rozsah na -1:1 golie rotation
        list_out[7] = list_coord[4] / 1.57  # mapuje rozsah na -1:1 def rotation
        list_out[8] = list_coord[5] / 1.57  # mapuje rozsah na -1:1 att rotation

        list_out[9] = (list_coord[6] * 4) - 0.8  # maps x coordinate of ball to -1:1 (1 close to att, -1 close to goal)
        list_out[10] = (list_coord[7] * 2.93255131964809) + 1  # maps y of ball to -1:1 left:right
        return list_out

    def show_gazebo_models(self):
        try:
            ball_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
            ball_coordinates_out = ball_coordinates('ball::ball', '')
            # print('\n')
            # print('Status.success = ', ball_coordinates_out.success)
            # print("ball X : " + str(ball_coordinates_out.link_state.pose.position.x))
            # print("ball Y : " + str(ball_coordinates_out.link_state.pose.position.y))

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
        if self.shot_iterator < len(pos_rot):
            env.publish_data(pos_rot[self.shot_iterator], pos_slide[self.shot_iterator], 0,
                             coord_list_out[1], 0, coord_list_out[0])
        else:
            #env.publish_data(0, 0, coord_list_out[3], coord_list_out[1], coord_list_out[2], coord_list_out[0])
            env.publish_data(0, 0, 0,coord_list_out[1], 0, coord_list_out[0])
        # Wait for period
        env.rate.sleep()
        # Get positions of players and ball
        [self.eval_data, self.state] = self.show_gazebo_models()
        # Reduce shower length by 1 second
        self.game_length -= 1

        # Calculate reward
        # Epsilon for overlap of players
        eps = 0.007
        # Penalty for moving
        action_penalty = abs(self.last_action[0] - action[0]) + abs(self.last_action[1] - action[1])# + \
                         #abs(self.last_action[2] - action[2]) + abs(self.last_action[3] - action[3]) #penalty rotation

        reward = 0 - action_penalty
        # Reward if players are in front of ball
        if self.eval_data[0] + eps <= self.eval_data[10] <= self.eval_data[0] - eps:
            reward += 1
        if self.eval_data[1] + eps <= self.eval_data[10] <= self.eval_data[1] - eps:
            reward += 1
        if self.eval_data[2] + eps <= self.eval_data[10] <= self.eval_data[2] - eps:
            reward += 1

        # if 0.5 < self.eval_data[6] < -0.5: #penalty velka rotace
        #     reward -= 10
        # if 0.5 < self.eval_data[7] < -0.5:
        #     reward -= 10

        # Check if game is done
        if self.eval_data[9] > 1:  # ball pos za utocnikama
            done = True
            reward += 50
        elif self.eval_data[9] < -1.05:  # ball pos gol
            done = True
            reward -= 500
        elif self.game_length == 0:
            done = True
        else:
            done = False
        if self.game_length == 0:
            done = True

        self.shot_iterator += 1
        self.last_action = action
        # Set placeholder for info
        info = {}

        # Return step information
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
        self.game_length = 60
        self.shot_iterator = 0
        return self.state


if __name__ == '__main__':
    # ROS node init
    rospy.wait_for_service('/gazebo/pause_physics')
    pause_physics_client = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    rospy.wait_for_service('/gazebo/unpause_physics')
    unpause_physics_client = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    unpause_physics_client()

    env = FutEnv()
    #check_env(env, warn=True)

    log_path = os.path.join('Trainingfut', 'LogsFut')

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    # Train the agent
    model.learn(total_timesteps=500000)
    # Save the agent
    model.save("PPO_NoRotation")

    pause_physics_client()

    # rospy.wait_for_service('/gazebo/reset_simulation')
    # reset_world = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
    # reset_world()
    # model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    # # Train the agent
    # model.learn(total_timesteps=50000)
    # # Save the agent
    # model.save("A2C_tst2")
