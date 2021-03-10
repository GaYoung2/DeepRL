import os
from time import sleep
import cv2
from rl_model import RlModel
from airsim_client import *
import datetime
import copy
import json

class DistributedAgent():
    def __init__(self):
        self.__model_buffer = None
        self.__model = None
        self.__airsim_started = False
        self.__data_dir = 'data/'
        self.__handle_dir = 'data/handle_image/'
        self.__per_iter_epsilon_reduction = 0.003
        self.__min_epsilon = 0.1
        self.__max_epoch_runtime_sec = float(30)
        self.__replay_memory_size = 50
        self.__batch_size = 32
        self.__experiment_name = 'local_run'
        self.__train_conv_layers = False
        self.__epsilon = 1
        self.__percent_full = 0
        self.__num_batches_run = 0
        self.__last_checkpoint_batch_count = 0
        self.__handles = {}
        self.__batch_update_frequency = 10
        self.__weights_path = None    
        self.__airsim_path = '../AD_Cookbook_AirSim/'
        self.__local_run = True
        self.__car_client = None
        self.__car_controls = None
        # self.__minibatch_dir = os.path.join(self.__data_dir, 'minibatches')
        # self.__output_model_dir = os.path.join(self.__data_dir, 'models')
        # self.__make_dir_if_not_exist(self.__minibatch_dir)
        # self.__make_dir_if_not_exist(self.__output_model_dir)
        self.__last_model_file = ''
        self.__possible_ip_addresses = []
        self.__trainer_ip_address = None
        self.__experiences = {}
        self.prev_steering = 0
        self.__init_road_points()
        self.__init_reward_points()
        self.__init_handle_images()
        self.__best_drive = datetime.timedelta(seconds=-1)  #added 2021-03-09 by kang
        self.__best_model = None  #added 2021-03-09 by kang
        self.__num_of_trial = 0

    def start(self):
        self.__run_function()

    def __run_function(self):
        self.__model = RlModel(self.__weights_path, self.__train_conv_layers)
        self.__connect_to_airsim()
        print('Filling replay memory...')
        while True:
            print('Running Airsim Epoch.')
            try:
                if self.__percent_full < 100:
                    self.__run_airsim_epoch(True)
                    self.__percent_full = 100.0 * len(self.__experiences['actions'])/self.__replay_memory_size
                    print('Replay memory now contains {0} members. ({1}% full)'.format(len(self.__experiences['actions']), self.__percent_full))
                else:
                    if (self.__model is not None):
                        print('Running Airsim Epoch.')
                        experiences, frame_count, drive_time = self.__run_airsim_epoch(False)
                        # If we didn't immediately crash, train on the gathered experiences
                        if (frame_count > 0):
                            print('Generating {0} minibatches...'.format(frame_count))
                            print('Sampling Experiences.')
                            # Sample experiences from the replay memory
                            sampled_experiences = self.__sample_experiences(experiences, frame_count, True)

                            self.__num_batches_run += frame_count
                            
                            # If we successfully sampled, train on the collected minibatches and send the gradients to the trainer node
                            if (len(sampled_experiences) > 0):
                                print('Publishing AirSim Epoch.')
                                self.__publish_batch_and_update_model(sampled_experiences, frame_count, drive_time)
                
            except msgpackrpc.error.TimeoutError:
                print('Lost connection to AirSim while fillling replay memory. Attempting to reconnect.')
                self.__connect_to_airsim()

    def __connect_to_airsim(self):
        attempt_count = 0
        while True:
            try:
                print('Attempting to connect to AirSim (attempt {0})'.format(attempt_count))
                self.__car_client = CarClient()
                self.__car_client.confirmConnection()
                self.__car_client.enableApiControl(True)
                self.__car_controls = CarControls()
                print('Connected!')
                return
            except:
                print('Failed to connect.')
                attempt_count += 1
                if (attempt_count % 10 == 0):
                    print('10 consecutive failures to connect. Attempting to start AirSim on my own.')
                    os.system('START "" powershell.exe {0}'.format(os.path.join(self.__airsim_path, 'AD_Cookbook_Start_AirSim.ps1 neighborhood -windowed')))
                print('Waiting a few seconds.')
                time.sleep(10)

    def __run_airsim_epoch(self, always_random):
        starting_points, starting_direction = self.__get_next_starting_point()
        # state_buffer_len = 4 changed by kang 2021-03-09 cuz of no use
        state_buffer = []
        wait_delta_sec = 0.01
        self.__car_client.simSetPose(Pose(Vector3r(starting_points[0], starting_points[1], starting_points[2]), AirSimClientBase.toQuaternion(starting_direction[0], starting_direction[1], starting_direction[2])), True)
        self.__car_controls.steering = 0
        self.__car_controls.throttle = 1
        self.__car_controls.brake = 0
        self.prev_steering = 0
        self.__car_client.setCarControls(self.__car_controls)
        time.sleep(1.5)
        state_buffer = self.__get_image()

        done = False
        actions = []
        pre_states = []
        post_states = []
        rewards = []
        predicted_rewards = []
        car_state = self.__car_client.getCarState()

        start_time = datetime.datetime.utcnow()
        end_time = start_time + datetime.timedelta(seconds=self.__max_epoch_runtime_sec)
        
        num_random = 0
        far_off = False

        while not done:
            collision_info = self.__car_client.getCollisionInfo()
            utc_now = datetime.datetime.utcnow()
            
            if (collision_info.has_collided or car_state.speed < 1 or utc_now > end_time or far_off):
                print('Start time: {0}, end time: {1}'.format(start_time, utc_now), file=sys.stderr)
                self.__car_controls.steering = 0
                self.__car_controls.throttle = 0
                self.__car_controls.brake = 1
                self.__car_client.setCarControls(self.__car_controls)
                time.sleep(4)
                if (utc_now > end_time):
                    print('timed out.')
                    print('Full autonomous run finished at {0}'.format(utc_now), file=sys.stderr)
                done = True
                sys.stderr.flush()
            else:
                # The Agent should occasionally pick random action instead of best action
                do_greedy = np.random.random_sample()
                pre_state = copy.deepcopy(state_buffer)
                angle = -int(self.prev_steering/0.05*4)
                pre_handle = self.__handles[angle].reshape(59,255,1)
                pre_state = np.concatenate([pre_state, pre_handle], axis=2)
                if (do_greedy < self.__epsilon or always_random):
                    num_random += 1
                    next_state = self.__model.get_random_state()
                    predicted_reward = 0
                else:
                    next_state, predicted_reward = self.__model.predict_state(pre_state)
                    print('Model predicts {0}'.format(next_state))
                # Convert the selected state to a control signal
                next_control_signals = self.__model.state_to_control_signals(next_state, self.__car_client.getCarState())

                # Take the action
                self.__car_controls.steering = self.prev_steering + next_control_signals[0]
                if self.__car_controls.steering > 1.0:
                    self.__car_controls.steering = 1.0
                elif self.__car_controls.steering < -1.0:
                    self.__car_controls.steering = -1.0
                self.prev_steering = self.__car_controls.steering
                print('change steering : ', self.prev_steering)
                self.__car_controls.throttle = next_control_signals[1]
                self.__car_controls.brake = next_control_signals[2]
                self.__car_client.setCarControls(self.__car_controls)
                
                # Wait for a short period of time to see outcome
                time.sleep(wait_delta_sec)

                # Observe outcome and compute reward from action
                state_buffer = self.__get_image()
                angle = -int(self.prev_steering/0.05*4)
                post_handle = self.__handles[angle].reshape(59,255,1)
                post_state = np.concatenate([state_buffer, post_handle],axis=2)
                car_state = self.__car_client.getCarState()
                collision_info = self.__car_client.getCollisionInfo()
                reward, far_off = self.__compute_reward(collision_info, car_state)
                
                # Add the experience to the set of examples from this iteration
                pre_states.append(pre_state)
                post_states.append(post_state)
                rewards.append(reward)
                predicted_rewards.append(predicted_reward)
                actions.append(next_state)
        # action수가 너무 적을경우, 그 회차의 학습을 진행하지 않음. #added 2021-03-09 by kang
        if len(actions) < 10:
            return self.__experiences, 0, 0

        is_not_terminal = [1 for i in range(0, len(actions)-1, 1)]
        is_not_terminal.append(0)
        self.__add_to_replay_memory('pre_states', pre_states)
        self.__add_to_replay_memory('post_states', post_states)
        self.__add_to_replay_memory('actions', actions)
        self.__add_to_replay_memory('rewards', rewards)
        self.__add_to_replay_memory('predicted_rewards', predicted_rewards)
        self.__add_to_replay_memory('is_not_terminal', is_not_terminal)

        print('Percent random actions: {0}'.format(num_random / max(1, len(actions))))
        print('Num total actions: {0}'.format(len(actions)))
        
        if not always_random:
            self.__epsilon -= self.__per_iter_epsilon_reduction
            self.__epsilon = max(self.__epsilon, self.__min_epsilon)
        
        return self.__experiences, len(actions), utc_now - start_time

    def __add_to_replay_memory(self, field_name, data):
        if field_name not in self.__experiences:
            self.__experiences[field_name] = data
        else:
            self.__experiences[field_name] += data
            start_index = max(0, len(self.__experiences[field_name]) - self.__replay_memory_size)
            self.__experiences[field_name] = self.__experiences[field_name][start_index:]

    def __sample_experiences(self, experiences, frame_count, sample_randomly):
        sampled_experiences = {}
        sampled_experiences['pre_states'] = []
        sampled_experiences['post_states'] = []
        sampled_experiences['actions'] = []
        sampled_experiences['rewards'] = []
        sampled_experiences['predicted_rewards'] = []
        sampled_experiences['is_not_terminal'] = []
        # Compute the surprise factor, which is the difference between the predicted an the actual Q value for each state.
        # We can use that to weight examples so that we are more likely to train on examples that the model got wrong.
        suprise_factor = np.abs(np.array(experiences['rewards'], dtype=np.dtype(float)) - np.array(experiences['predicted_rewards'], dtype=np.dtype(float)))
        suprise_factor_normalizer = np.sum(suprise_factor)
        suprise_factor /= float(suprise_factor_normalizer)

        # Generate one minibatch for each frame of the run
        for _ in range(0, frame_count, 1):
            if sample_randomly:
                idx_set = set(np.random.choice(list(range(0, suprise_factor.shape[0], 1)), size=(self.__batch_size), replace=False))
            else:
                idx_set = set(np.random.choice(list(range(0, suprise_factor.shape[0], 1)), size=(self.__batch_size), replace=False, p=suprise_factor))
        
            sampled_experiences['pre_states'] += [experiences['pre_states'][i] for i in idx_set]
            sampled_experiences['post_states'] += [experiences['post_states'][i] for i in idx_set]
            sampled_experiences['actions'] += [experiences['actions'][i] for i in idx_set]
            sampled_experiences['rewards'] += [experiences['rewards'][i] for i in idx_set]
            sampled_experiences['predicted_rewards'] += [experiences['predicted_rewards'][i] for i in idx_set]
            sampled_experiences['is_not_terminal'] += [experiences['is_not_terminal'][i] for i in idx_set]
            
        return sampled_experiences
        
    def __publish_batch_and_update_model(self, batches, batches_count, drive_time): # added 2021-03-09 by kang
        # Train and get the gradients
        print('Publishing epoch data and getting latest model from parameter server...')
        gradients = self.__model.get_gradient_update_from_batches(batches) 
    
        if (self.__num_batches_run > self.__batch_update_frequency + self.__last_checkpoint_batch_count):
            self.__model.update_critic()
            
            checkpoint = {}
            checkpoint['model'] = self.__model.to_packet(get_target=True)
            checkpoint['batch_count'] = batches_count
            checkpoint_str = json.dumps(checkpoint)

            checkpoint_dir = os.path.join(os.path.join(self.__data_dir, 'checkpoint'), self.__experiment_name)
            
            if not os.path.isdir(checkpoint_dir):
                try:
                    os.makedirs(checkpoint_dir)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                        
            file_name = os.path.join(checkpoint_dir,'{0}.json'.format(self.__num_batches_run)) 
            with open(file_name, 'w') as f:
                print('Checkpointing to {0}'.format(file_name))
                f.write(checkpoint_str)
            
            self.__last_checkpoint_batch_count = self.__num_batches_run
            
            # 운행시간을 이용해서 가장 오래 걸린 시간을 best policy로 보고, best policy를 따로 저장. #added 2021-03-09 by kang
            if drive_time > self.__best_drive:
                print("="*30)
                print("New Best Policy!!!!!!")
                print("="*30)
                self.__best_drive = drive_time
                bestpoint_dir = os.path.join(os.path.join(self.__data_dir, 'bestpoint'), self.__experiment_name)

                if not os.path.isdir(bestpoint_dir):
                    try:
                        os.makedirs(bestpoint_dir)
                    except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise
                file_name = os.path.join(bestpoint_dir,'{0}.json'.format(self.__num_batches_run)) 

                with open(file_name, 'w') as f:
                    print('Add Best Policy to {0}'.format(file_name))
                    f.write(checkpoint_str)
                self.__best_model = self.__model
                self.__best_experiences = self.__experiences
                
            # for test store best policy
            # elif self.__num_of_trial > 10:
            #     print("="*30)
            #     print("Reload best Model")
            #     print("="*30)
            #     self.__model = self.__best_model
            #     self.__experiences = self.__best_experiences
            #     self.__num_of_trial = 0
            # self.__num_of_trial += 1

    def __compute_reward(self, collision_info, car_state):
        #Define some constant parameters for the reward function
        THRESH_DIST = 3.5                # The maximum distance from the center of the road to compute the reward function
        DISTANCE_DECAY_RATE = 1.2        # The rate at which the reward decays for the distance function
        CENTER_SPEED_MULTIPLIER = 2.0    # The ratio at which we prefer the distance reward to the speed reward

        # If the car has collided, the reward is always zero
        # 충돌 시 reward를 음수로 줘보았음.
        if (collision_info.has_collided):
            return -0.5, True
        
        # If the car is stopped, the reward is always zero
        speed = car_state.speed
        if (speed < 2):
            return 0.0, True
        
        #Get the car position
        position_key = bytes('position', encoding='utf8')
        x_val_key = bytes('x_val', encoding='utf8')
        y_val_key = bytes('y_val', encoding='utf8')

        car_point = np.array([car_state.kinematics_true[position_key][x_val_key], car_state.kinematics_true[position_key][y_val_key], 0])
        
        # Distance component is exponential distance to nearest line
        distance = 999
        
        #Compute the distance to the nearest center line
        for line in self.__reward_points:
            local_distance = 0
            length_squared = ((line[0][0]-line[1][0])**2) + ((line[0][1]-line[1][1])**2)
            if (length_squared != 0):
                t = max(0, min(1, np.dot(car_point-line[0], line[1]-line[0]) / length_squared))
                proj = line[0] + (t * (line[1]-line[0]))  
                local_distance = np.linalg.norm(proj - car_point)
            
            distance = min(local_distance, distance)
            
        distance_reward = math.exp(-(distance * DISTANCE_DECAY_RATE))
        
        return distance_reward, distance > THRESH_DIST

    def __get_image(self):
        image_response = self.__car_client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
        image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
        image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
        image_rgba = image_rgba[76:135,0:255,0:3].astype(float)
        image_rgba = image_rgba.reshape(59, 255, 3)
        return image_rgba

    def __init_road_points(self):
        self.__road_points = []
        car_start_coords = [12961.722656, 6660.329102, 0]
        with open(os.path.join(os.path.join(self.__data_dir, 'data'), 'road_lines.txt'), 'r') as f:
            for line in f:
                points = line.split('\t')
                first_point = np.array([float(p) for p in points[0].split(',')] + [0])
                second_point = np.array([float(p) for p in points[1].split(',')] + [0])
                self.__road_points.append(tuple((first_point, second_point)))

        # Points in road_points.txt are in unreal coordinates
        # But car start coordinates are not the same as unreal coordinates
        for point_pair in self.__road_points:
            for point in point_pair:
                point[0] -= car_start_coords[0]
                point[1] -= car_start_coords[1]
                point[0] /= 100
                point[1] /= 100
              
    def __init_reward_points(self):
        self.__reward_points = []
        with open(os.path.join(os.path.join(self.__data_dir, 'data'), 'reward_points.txt'), 'r') as f:
            for line in f:
                point_values = line.split('\t')
                first_point = np.array([float(point_values[0]), float(point_values[1]), 0])
                second_point = np.array([float(point_values[2]), float(point_values[3]), 0])
                self.__reward_points.append(tuple((first_point, second_point)))

    def __get_next_starting_point(self):
    
        # Get the current state of the vehicle
        car_state = self.__car_client.getCarState()

        # Pick a random road.
        random_line_index = np.random.randint(0, high=len(self.__road_points))
        
        # Pick a random position on the road. 
        # Do not start too close to either end, as the car may crash during the initial run.
        
        random_interp = 0.15    # changed by GY 21-03-10
        
        # Pick a random direction to face
        random_direction_interp = 0.4 # changed by GY 21-03-10

        # Compute the starting point of the car
        random_line = self.__road_points[random_line_index]
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

    def __init_handle_images(self):
        self.__handles = {0 : cv2.cvtColor(cv2.imread(self.__handle_dir+'0.png'), cv2.COLOR_BGR2GRAY),
                        20 : cv2.cvtColor(cv2.imread(self.__handle_dir+'right20.png'), cv2.COLOR_BGR2GRAY),
                        40 : cv2.cvtColor(cv2.imread(self.__handle_dir+'right40.png'), cv2.COLOR_BGR2GRAY),
                        60 : cv2.cvtColor(cv2.imread(self.__handle_dir+'right60.png'), cv2.COLOR_BGR2GRAY),
                        80 : cv2.cvtColor(cv2.imread(self.__handle_dir+'right80.png'), cv2.COLOR_BGR2GRAY),
                        -20 : cv2.cvtColor(cv2.imread(self.__handle_dir+'left20.png'), cv2.COLOR_BGR2GRAY),
                        -40 : cv2.cvtColor(cv2.imread(self.__handle_dir+'left40.png'), cv2.COLOR_BGR2GRAY),
                        -60 : cv2.cvtColor(cv2.imread(self.__handle_dir+'left60.png'), cv2.COLOR_BGR2GRAY),
                        -80 : cv2.cvtColor(cv2.imread(self.__handle_dir+'left80.png'), cv2.COLOR_BGR2GRAY)}
agent = DistributedAgent()
agent.start()