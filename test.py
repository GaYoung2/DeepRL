from airsim_client import *
import numpy as np
import time

print('Connecting to AirSim...')
car_client = CarClient()
car_client.confirmConnection()
car_controls = CarControls()
car_client.setCarControls(car_controls)
print('Connected!')

def get_image(car_client):
    image_response = car_client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    image_rgba = image_rgba[76:135,0:255,0:3].astype(float)
    image_rgba = image_rgba.reshape(59, 255, 3)
    return image_rgba

#added by wb -> change starting point
#---------------------------------------
def get_next_starting_point(car_client):
    
        # Get the current state of the vehicle
        car_state = car_client.getCarState()

        # Pick a random road.
        random_line_index = np.random.randint(0, high=len(road_points))
        
        # Pick a random position on the road. 
        # Do not start too close to either end, as the car may crash during the initial run.
        
        random_interp = 0.15    # changed by GY 21-03-10
        
        # Pick a random direction to face
        random_direction_interp = 0.4 # changed by GY 21-03-10

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
    with open(os.path.join(os.path.join('data/', 'data'), 'road_lines.txt'), 'r') as f:
        for line in f:
            points = line.split('\t')
            first_point = np.array([float(p) for p in points[0].split(',')] + [0])
            second_point = np.array([float(p) for p in points[1].split(',')] + [0])
            road_points.append(tuple((first_point, second_point)))
    for point_pair in road_points:
        for point in point_pair:
            point[0] -= car_start_coords[0]
            point[1] -= car_start_coords[1]
            point[0] /= 100
            point[1] /= 100
    return road_points

road_points = init_road_points() 
#starting_points, starting_direction = get_next_starting_point(car_client)
#car_client.simSetPose(Pose(Vector3r(starting_points[0], starting_points[1], starting_points[2]), AirSimClientBase.toQuaternion(starting_direction[0], starting_direction[1], starting_direction[2])), True)
car_state = car_client.getCarState()
orientation_key = bytes('orientation', encoding='utf8')
x_val_key = bytes('x_val', encoding='utf8')
y_val_key = bytes('y_val', encoding='utf8')
z_val_key = bytes('z_val', encoding='utf8')
w_val_key = bytes('w_val', encoding='utf8')
print(car_state.kinematics_true)
