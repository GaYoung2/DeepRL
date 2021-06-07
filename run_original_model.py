from airsim_client import *
from original_model import RlModel
import numpy as np
import time
import sys
import json
import PIL
import PIL.ImageFilter
import datetime

MODEL_FILENAME = 'data/bestpoint/original+random/39130.json' #Your model goes here
model = RlModel(None, False)
with open(MODEL_FILENAME, 'r') as f:
    checkpoint_data = json.loads(f.read())
    model.from_packet(checkpoint_data['model'])

print('Connecting to AirSim...')
car_client = CarClient()
car_client.confirmConnection()
car_client.enableApiControl(True)
car_controls = CarControls()
print('Connected!')

def get_image(car_client):
    image_response = car_client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)

    return image_rgba[76:135,0:255,0:3].astype(float)

def append_to_ring_buffer(item, buffer, buffer_size):
    if (len(buffer) >= buffer_size):
        buffer = buffer[1:]
    buffer.append(item)
    return buffer
def get_next_starting_point(car_client):

    # Get the current state of the vehicle
    car_state = car_client.getCarState()
    global random_line_index
    # Pick a random road.
    random_line_index = np.random.randint(0, high=len(road_points))
    random_line_index = 1
    # Pick a random position on the road. 
    # Do not start too close to either end, as the car may crash during the initial run.
    
    # added return to origin by Kang 21-03-10
    random_interp = 0.5    # changed by GY 21-03-10

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
    road = 'road_lines.txt'
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

road_points = init_road_points() 
starting_points, starting_direction = get_next_starting_point(car_client)
car_client.simSetPose(Pose(Vector3r(starting_points[0], starting_points[1], starting_points[2]), AirSimClientBase.toQuaternion(starting_direction[0], starting_direction[1], starting_direction[2])), True)
#---------------------------------------
state_buffer = []
state_buffer_len = 4

print('Running car for a few seconds...')
car_controls.steering = 0
car_controls.throttle = 1
car_controls.brake = 0
car_client.setCarControls(car_controls)
stop_run_time =datetime.datetime.now() + datetime.timedelta(seconds=2)
while(datetime.datetime.now() < stop_run_time):
    time.sleep(0.01)
    state_buffer = append_to_ring_buffer(get_image(car_client), state_buffer, state_buffer_len)

print('Running model')
while(True):
    state_buffer = append_to_ring_buffer(get_image(car_client), state_buffer, state_buffer_len)
    next_state, dummy = model.predict_state(state_buffer)
    next_control_signal = model.state_to_control_signals(next_state, car_client.getCarState())

    car_controls.steering = next_control_signal[0]
    car_controls.throttle = next_control_signal[1]
    car_controls.brake = next_control_signal[2]

    print('State = {0}, steering = {1}, throttle = {2}, brake = {3}'.format(next_state, car_controls.steering, car_controls.throttle, car_controls.brake))

    car_client.setCarControls(car_controls)

    time.sleep(0.1)