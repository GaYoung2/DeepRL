from Share.scripts_downpour.app.airsim_client import *
import numpy as np
import time
import sys
import json
import matplotlib.pyplot as plt
from IPython.display import clear_output #print값을 마지막 값만 print할 수 있도록 해줌
import time
import PIL
import PIL.ImageFilter

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, clone_model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam, Adagrad, Adadelta
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
import keras.backend as K
from keras.preprocessing import image
from keras.initializers import random_normal


def compute_reward(car_state, collision_info, road_points):
#보상함수(자동차가 도로의 중심에 가까울 수록 높은 보상, 보상 범위 [0,1])

    #Define some constant parameters for the reward function
    THRESH_DIST = 3.5   #자동차와 도로 중심 간의 최대 거리
    # The maximum distance from the center of the road to compute the reward function
    DISTANCE_DECAY_RATE = 1.2 #거리 함수에 대한 보상이 소멸하는 속도   
    # The rate at which the reward decays for the distance function
    CENTER_SPEED_MULTIPLIER = 2.0 #속도 보상보다 거리 보상을 선호하는 비율 
    # The ratio at which we prefer the distance reward to the speed reward
    
    # If the car is stopped, the reward is always zero
    #차가 멈추면 언제나 보상은 0이다.
    speed = car_state.speed #자동차 상태로부터 스피드를 받아온다.
    if (speed < 2): #속도<2==자동차가 멈춤
        return 0
    
    #Get the car position 자동차의 위치(중앙으로부터 떨어진 정도)
    position_key = bytes('position', encoding='utf8') #position_key에는 b'position_key'가 들어가게 된다.
    x_val_key = bytes('x_val', encoding='utf8')
    y_val_key = bytes('y_val', encoding='utf8')

    car_point = np.array([car_state.kinematics_true[position_key][x_val_key], car_state.kinematics_true[position_key][y_val_key], 0])
    
    # Distance component is exponential distance to nearest line
    distance = 999 #가장 가까운 차선에 대한 거리
    
    #Compute the distance to the nearest center line
    #차선과의 거리를 계산하는 방법
    for line in road_points:
    #road_points=[(array([-251.21722656, -209.60329102,0]), array([-132.21722656, -209.60329102,0]))...]
        local_distance = 0
        length_squared = ((line[0][0]-line[1][0])**2) + ((line[0][1]-line[1][1])**2) 
        #차선중앙과 차와의 거리의 제곱
        #point[0][0]은 차의 x축 위치, point[0][1]은 차의 y축 위치, point[0][2]은 차의 z축 위치(차는 0, 드론은 있음)
        #point[0][0]은 가장 가까운 선의 x축 위치, point[0][1]은 가장 가까운 선의 y축 위치, point[0][2]은 가장 가까운 선의 z축 위치(차는 0, 드론은 있음)
        if (length_squared != 0): 
            t = max(0, min(1, np.dot(car_point-line[0], line[1]-line[0]) / length_squared))
            #np.dot은 numpy array를 곱할 때 사용한다.
            proj = line[0] + (t * (line[1]-line[0]))
            local_distance = np.linalg.norm(proj - car_point)
        
        distance = min(distance, local_distance)
        
    distance_reward = math.exp(-(distance * DISTANCE_DECAY_RATE))
    
    return distance_reward


plt.figure()
   
# Reads in the reward function lines
def init_reward_points():
    road_points = [] #road_points가 배열
    with open('Share\\data\\reward_points.txt', 'r') as f: 
        #'Share\\data\\reward_points.txt'를 가면, rewardpoint가 1줄에 4개의 value가 나열되어있다.
        for line in f: #한줄한줄 읽기
            point_values = line.split('\t') #예를 들어 [-251.21722655999997,-209.60329102,-132.21722655999997,-209.60329102]
            first_point = np.array([float(point_values[0]), float(point_values[1]), 0])
            #예를들어 array([-251.21722656, -209.60329102,0]
            second_point = np.array([float(point_values[2]), float(point_values[3]), 0])
            #예를 들어 array[-132.21722655999997,-209.60329102,0]
            road_points.append(tuple((first_point, second_point)))
            #road point = [(array([-251.21722656, -209.60329102,0]), array([-132.21722656, -209.60329102,0]))]

    return road_points

#Draws the car location plot
def draw_rl_debug(car_state, road_points): #그림 그려주는 함수
    fig = plt.figure(figsize=(15,15))
    print('')
    for point in road_points:
        plt.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]], 'k-', lw=2)
    position_key = bytes('position', encoding='utf8')
    x_val_key = bytes('x_val', encoding='utf8')
    y_val_key = bytes('y_val', encoding='utf8')
    
    car_point = np.array([car_state.kinematics_true[position_key][x_val_key], car_state.kinematics_true[position_key][y_val_key], 0])
    plt.plot([car_point[0]], [car_point[1]], 'bo')
    
    plt.show()
    
reward_points = init_reward_points()
#reward_points를 numpy모양으로 다 받아온다.
    
car_client = CarClient() #Share.scripts_downpour.app.airsim_client에서 CarClient를 받아온다.
#이걸 만들면 ip="127.0.0.1"
car_client.confirmConnection()
#airsim과 연결 기다리기
car_client.enableApiControl(False)

try:
    while(True):
        clear_output(wait=True) #print문을 줄여줌(마지막만 프린트해준다.)
        car_state = car_client.getCarState() #자동차의 현재 state를 받아옴
        collision_info = car_client.getCollisionInfo() #자동차의 현재 collisionInfo를 받아옴
        reward = compute_reward(car_state, collision_info, reward_points)
        print('Current reward: {0:.2f}'.format(reward))
        draw_rl_debug(car_state, reward_points)
        time.sleep(1)

#Handle interrupt gracefully
except:
    pass