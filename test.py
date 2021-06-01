from six import b
from airsim_client import *
import numpy as np
import time

random_respawn = False
print('Connecting to AirSim...')
car_client = CarClient()
car_client.confirmConnection()
car_controls = CarControls()
car_client.setCarControls(car_controls)
print('Connected!')
random_line_index = -1

def get_image(car_client):
    image_response = car_client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    image_rgba = image_rgba[76:135,0:255,0:3].astype(float)
    image_rgba = image_rgba.reshape(59, 255, 3)
    return image_rgba

def get_next_starting_point(car_client):

    # Get the current state of the vehicle
    car_state = car_client.getCarState()
    global random_line_index
    # Pick a random road.
    random_line_index = np.random.randint(0, high=len(road_points))
    
    # Pick a random position on the road. 
    # Do not start too close to either end, as the car may crash during the initial run.
    
    # added return to origin by Kang 21-03-10
    random_interp = 0.1    # changed by GY 21-03-10

    # Pick a random direction to face
    random_direction_interp = 0.9 # changed by GY 21-03-10

    # Compute the starting point of the car
    random_line = road_points[random_line_index]
    random_start_point = list(random_line[0])
    random_start_point[0] += (random_line[1][0] - random_line[0][0])*random_interp
    random_start_point[1] += (random_line[1][1] - random_line[0][1])*random_interp

    # Compute the direction that the vehicle will face
    # Vertical line
    if (np.isclose(random_line[0][1], random_line[1][1])):
        if (random_direction_interp > 0.5):
            random_direction = (0,0,0)
        else:
            random_direction = (0, 0, math.pi)
    # Horizontal line
    elif (np.isclose(random_line[0][0], random_line[1][0])):
        if (random_direction_interp > 0.5):
            random_direction = (0,0,math.pi/2)
        else:
            random_direction = (0,0,-1.0 * math.pi/2)

    # The z coordinate is always zero
    random_start_point[2] = -0
    return (random_start_point, random_direction)

def init_road_points():
    road_points = []
    car_start_coords = [12961.722656, 6660.329102, 0]
    road = ''
    if not random_respawn:
        road = 'road_lines.txt'
    else:
        road = 'origin_road_lines.txt'
    with open(os.path.join(os.path.join('data', 'data'), road), 'r') as f:
        for line in f:
            points = line.split('\t')
            first_point = np.array([float(p) for p in points[0].split(',')] + [0])
            second_point = np.array([float(p) for p in points[1].split(',')] + [0])
            road_points.append(tuple((first_point, second_point)))

    # Points in road_points.txt are in unreal coordinates
    # But car start coordinates are not the same as unreal coordinates
    for point_pair in road_points:
        for point in point_pair:
            point[0] -= car_start_coords[0]
            point[1] -= car_start_coords[1]
            point[0] /= 100
            point[1] /= 100
    return road_points


def compute_reward(self, collision_info, car_state):
        #Define some constant parameters for the reward function
        THRESH_DIST = 4.0                # The maximum distance from the center of the road to compute the reward function
        DISTANCE_DECAY_RATE = 1.2        # The rate at which the reward decays for the distance function
        CENTER_SPEED_MULTIPLIER = 2.0    # The ratio at which we prefer the distance reward to the speed reward

        # If the car has collided, the reward is always zero
        if (collision_info.has_collided):
            return 0.0, True
        
        # If the car is stopped, the reward is always zero
        speed = car_state.speed
        if (speed < 2 or collision_info==True):
            return 0.0, True
        
        #Get the car position
        position_key = bytes('position', encoding='utf8') #position of x, y
        orientation_key = bytes('orientation', encoding='utf8') #direction
        x_val_key = bytes('x_val', encoding='utf8')
        y_val_key = bytes('y_val', encoding='utf8')
        z_val_key = bytes('z_val', encoding='utf8') 

        car_point = np.array([car_state.kinematics_true[position_key][x_val_key], car_state.kinematics_true[position_key][y_val_key], 0])
        
        # Distance component is exponential distance to nearest line
        distance = 999
        
        # Compute the distance to the nearest center line
        for line in self.__reward_points:
            local_distance = 0
            length_squared = ((line[0][0]-line[1][0])**2) + ((line[0][1]-line[1][1])**2)
            if (length_squared != 0):
                t = max(0, min(1, np.dot(car_point-line[0], line[1]-line[0]) / length_squared))
                proj = line[0] + (t * (line[1]-line[0]))  
                local_distance = np.linalg.norm(proj - car_point)
            
            distance = min(local_distance, distance)
        direction = 0

        if random_line_index == 0:
            if -5< car_state.kinematics_true[position_key][x_val_key] < 8 and -82 < car_state.kinematics_true[position_key][y_val_key] < 48:
                if -0.7 <= car_state.kinematics_true[orientation_key][z_val_key] <=0:
                    direction = (0.7 - car_state.kinematics_true[orientation_key][z_val_key])/1.4
                elif -1 <= car_state.kinematics_true[orientation_key][z_val_key] <=-0.7:
                    direction = (13/6 + 5*car_state.kinematics_true[orientation_key][z_val_key]/3)
                elif 0 <= car_state.kinematics_true[orientation_key][z_val_key] <=0.7:
                    direction = 0.5 - car_state.kinematics_true[orientation_key][z_val_key]*5/7
                else:
                    direction = 0
            elif -120< car_state.kinematics_true[position_key][x_val_key]  < 0 and 40 < car_state.kinematics_true[position_key][y_val_key] < 49:
                if 0 <= car_state.kinematics_true[orientation_key][z_val_key] <=0.7:
                    direction = 1 - 5*car_state.kinematics_true[orientation_key][z_val_key]/7
                elif -0.7 <= car_state.kinematics_true[orientation_key][z_val_key] <=0:
                    direction = 1 + 5*car_state.kinematics_true[orientation_key][z_val_key]/7
                else:
                    direction = 0
        else:
            if -120< car_state.kinematics_true[position_key][x_val_key]  < 0 and 40 < car_state.kinematics_true[position_key][y_val_key] < 49:
                if -1 <= car_state.kinematics_true[orientation_key][z_val_key] <=-0.7:
                    direction = -2/3 - 5*car_state.kinematics_true[orientation_key][z_val_key]/3
                elif 0.7 <= car_state.kinematics_true[orientation_key][z_val_key] <=1:
                    direction = -2/3 + 5*car_state.kinematics_true[orientation_key][z_val_key]/3
                elif 0 <= car_state.kinematics_true[orientation_key][z_val_key] <=0.7:
                    direction = car_state.kinematics_true[orientation_key][z_val_key]*5/7
                else:
                    direction = 0
            elif -5< car_state.kinematics_true[position_key][x_val_key] < 8 and -82 < car_state.kinematics_true[position_key][y_val_key] < 48:
                if 0.7 <= car_state.kinematics_true[orientation_key][z_val_key] <=1:
                    direction = 13/6 - car_state.kinematics_true[orientation_key][z_val_key]*5/3
                elif 0 <= car_state.kinematics_true[orientation_key][z_val_key] <=0.7:
                    direction = 0.5 + car_state.kinematics_true[orientation_key][z_val_key]*5/7
                else:
                    direction = 0

        reward = math.exp(-((distance+direction)/2 * DISTANCE_DECAY_RATE))
        
        return reward, distance > THRESH_DIST
# road_points = init_road_points() 
# print(road_points)
# starting_points, starting_direction = get_next_starting_point(car_client)
# print(starting_points)
# car_client.simSetPose(Pose(Vector3r(starting_points[0], starting_points[1], starting_points[2]), AirSimClientBase.toQuaternion(starting_direction[0], starting_direction[1], starting_direction[2])), True)
car_state = car_client.getCarState()
orientation_key = bytes('orientation', encoding='utf8')
position_key = bytes('position', encoding='utf8')
x_val_key = bytes('x_val', encoding='utf8')
y_val_key = bytes('y_val', encoding='utf8')
z_val_key = bytes('z_val', encoding='utf8')
w_val_key = bytes('w_val', encoding='utf8')
# print(car_state.kinematics_true[position_key][x_val_key])
random_line_index = 1
compute_reward(car_state)
