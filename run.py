from airsim_client import *
from rl_model import RlModel
import numpy as np
import time
import sys
import json
import PIL
import PIL.ImageFilter
import datetime
import cv2

#MODEL_FILENAME = 'sample_model.json' #Your model goes here
# MODEL_FILENAME = 'D:\checkpoint\local_run/13893.json'
MODEL_FILENAME = 'data/checkpoint/local_run/286238.json'
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
    image_rgba = image_rgba[76:135,0:255,0:3].astype(float)
    image_rgba = image_rgba.reshape(59, 255, 3)
    return image_rgba

def append_to_ring_buffer(item, buffer, buffer_size):
    if (len(buffer) >= buffer_size):
        buffer = buffer[1:]
    buffer.append(item)
    return buffer

state_buffer = []
state_buffer_len = 4

print('Running car for a few seconds...')
car_controls.steering = 0
car_controls.throttle = 1
car_controls.brake = 0
car_client.setCarControls(car_controls)
stop_run_time =datetime.datetime.now() + datetime.timedelta(seconds=1.5)
while(datetime.datetime.now() < stop_run_time):
    time.sleep(0.01)
prev_steering = 0
handle_dir = 'data/handle_image/'
handles = {0 : cv2.cvtColor(cv2.imread(handle_dir+'0.png'), cv2.COLOR_BGR2GRAY),
            20 : cv2.cvtColor(cv2.imread(handle_dir+'right20.png'), cv2.COLOR_BGR2GRAY),
            40 : cv2.cvtColor(cv2.imread(handle_dir+'right40.png'), cv2.COLOR_BGR2GRAY),
            60 : cv2.cvtColor(cv2.imread(handle_dir+'right60.png'), cv2.COLOR_BGR2GRAY),
            80 : cv2.cvtColor(cv2.imread(handle_dir+'right80.png'), cv2.COLOR_BGR2GRAY),
            -20 : cv2.cvtColor(cv2.imread(handle_dir+'left20.png'), cv2.COLOR_BGR2GRAY),
            -40 : cv2.cvtColor(cv2.imread(handle_dir+'left40.png'), cv2.COLOR_BGR2GRAY),
            -60 : cv2.cvtColor(cv2.imread(handle_dir+'left60.png'), cv2.COLOR_BGR2GRAY),
            -80 : cv2.cvtColor(cv2.imread(handle_dir+'left80.png'), cv2.COLOR_BGR2GRAY)}
print('Running model')
while(True):
    state_buffer = get_image(car_client)
    angle = -int(prev_steering/0.05*4)
    pre_handle = handles[angle].reshape(59,255,1)
    state_buffer = np.concatenate([state_buffer, pre_handle], axis=2)
    next_state, dummy = model.predict_state(state_buffer)
    next_control_signal = model.state_to_control_signals(next_state, car_client.getCarState())

    car_controls.steering = next_control_signal[0]
    car_controls.throttle = next_control_signal[1]
    car_controls.brake = next_control_signal[2]

    print('State = {0}, steering = {1}, throttle = {2}, brake = {3}'.format(next_state, car_controls.steering, car_controls.throttle, car_controls.brake))

    car_client.setCarControls(car_controls)

    time.sleep(0.1)