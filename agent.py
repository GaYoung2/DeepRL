import os
import shutil
from time import sleep
import cv2
from keras.engine.input_layer import Input
from keras.models import load_model
from rl_model import RlModel
from airsim_client import *
import datetime
import copy
import json
import pickle as pkl
from lane_detection import Lanes, road_lines

# 액션을 취하는 간격 약 0.05초
# airsim simulator에서는 속도단위는 m/s
# 액션 간격 x 속도 = 운행거리(m단위)

# for start from the model added by kang 21-03-10
# MODEL_FILENAME = 'data/saved_point/best_model.json'
MODEL_FILENAME = None

# load lane_net model
trained_model  = None
random_respawn = False
EXPERIENCE_FILENAME = 'latest.pkl'
class DistributedAgent():
    def __init__(self, use_handle=True, use_lane = True,use_speed = True):
        self.__model_buffer = None
        self.__model = None
        self.__lane_model = trained_model
        self.__lanes = Lanes()
        self.__airsim_started = False
        self.__data_dir = 'data/'
        self.__handle_dir = 'data/handle_image/'
        self.__airsim_path = '../AD_Cookbook_AirSim/'
        self.__per_iter_epsilon_reduction = 0.003
        self.__min_epsilon = 0.1
        self.__max_epoch_runtime_sec = float(50)
        self.__replay_memory_size = 50
        self.__batch_size = 32
        self.__experiment_name = 'deep+handle+speed+far_reward_penalty_1;9'
        self.__train_conv_layers = False
        self.__epsilon = 1
        self.__percent_full = 0
        self.__num_batches_run = 0
        self.__last_checkpoint_batch_count = 0
        self.__handles = {}
        self.__batch_update_frequency = 10
        self.__weights_path = None    
        self.__car_client = None
        self.__car_controls = None
        # self.__minibatch_dir = os.path.join(self.__data_dir, 'minibatches')
        # self.__output_model_dir = os.path.join(self.__data_dir, 'models')
        # self.__make_dir_if_not_exist(self.__minibatch_dir)
        # self.__make_dir_if_not_exist(self.__output_model_dir)
        self.__last_model_file = ''
        self.__experiences = {}
        self.prev_steering = 0
        self.__init_road_points()
        self.__init_reward_points()
        self.__init_handle_images()
        self.__best_drive = 0
        # self.__best_drive = datetime.timedelta(seconds=-1)  #added 2021-03-09 by kang
        ### removed by Kang 21-05-13 ###
        # self.__best_model = None  #added 2021-03-09 by kang
        # self.__best_epsilon = 1
        ################################
        self.__num_of_trial = 0
        self.__the_start_time = datetime.datetime.utcnow()
        self.__total_reward = 0
        self.__drive_time = 0
        ### 분기를 나누기 위해 더해진 3개의 변수들 ###
        self.__use_handle = use_handle
        self.__use_lane = use_lane
        self.__use_speed = use_speed
        ###########################################
        self.__drive_distance = 0   # reward에 지나간 거리 추가
        self.__prev_car_point = None
        
    def start(self):  
        self.__run_function()

    def __run_function(self):
        self.__model = RlModel(self.__weights_path, self.__train_conv_layers, run = False, use_handle=self.__use_handle, use_lane=self.__use_lane, use_speed=self.__use_speed)
        
        # Read json model file from here by Kang 21-03-10
        if MODEL_FILENAME != None:
            with open(os.path.join('data/saved_point/',MODEL_FILENAME), 'r') as f:
                checkpoint_data = json.loads(f.read())
                self.__model.from_packet(checkpoint_data['model'])
            print("Latest Model Loaded!")

            # peakle을 이용해서 self.__experiences 불러오기
            # saved_file = open(os.path.join('data/saved_point/',EXPERIENCE_FILENAME), 'rb')
            saved_file = open(EXPERIENCE_FILENAME, 'rb')
            loaded_file = pkl.load(saved_file)
            self.__experiences = loaded_file[0]
            self.__epsilon = loaded_file[1]
            self.__num_batches_run = loaded_file[2]
            saved_file.close()   

        self.__connect_to_airsim()
        
        while True:
            print('Running Airsim Epoch.')
            try:
                if self.__percent_full < 100 and MODEL_FILENAME == None:    # Model이 있을 경우에는 experiences와 epsilon을 불러오기 때문에 replay memory 채울필요 X
                    print('Filling replay memory...')
                    self.__run_airsim_epoch(MODEL_FILENAME == None)
                    try:
                        self.__percent_full = 100.0 * len(self.__experiences['actions'])/self.__replay_memory_size
                        print('Replay memory now contains {0} members. ({1}% full)'.format(len(self.__experiences['actions']), self.__percent_full))
                    except:
                        print('experience memory is empty, fill again.')

                else:
                    if (self.__model is not None):
                        # experiences, frame_count, self.__drive_time = self.__run_airsim_epoch(False)
                        experiences, frame_count = self.__run_airsim_epoch(False)
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
                                self.__publish_batch_and_update_model(sampled_experiences, frame_count, self.__drive_time)
                            self.__drive_distance = 0
                
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
        time.sleep(1.5)
        # move place up to get time to load lane detect model
        state_buffer, state_lane = self.__get_image(use_lane=self.__use_lane)
        
        if self.__use_lane:
            state_lane.shape = (59,255,1)
            state_buffer = np.concatenate([state_buffer, state_lane], axis=2)

        if self.__use_handle:
            post_handle = self.__handles[0].reshape(59,255,1)
            cv2.imshow('handle',post_handle)

            state_buffer = np.concatenate([state_buffer, post_handle], axis=2)
        
        if self.__use_speed:
            self.__car_controls.throttle = 0
        else:
            self.__car_controls.throttle = 0.5
        self.__car_controls.brake = 0
        self.prev_steering = 0
        self.__car_client.setCarControls(self.__car_controls)
        time.sleep(1.5)
        done = False
        actions = []
        throttle = []
        pre_states = []
        post_states = []
        rewards = []
        predicted_rewards = []
        car_state = self.__car_client.getCarState()
        #Get the car position
        position_key = bytes('position', encoding='utf8')
        x_val_key = bytes('x_val', encoding='utf8')
        y_val_key = bytes('y_val', encoding='utf8')
        self.__prev_car_point =  np.array([car_state.kinematics_true[position_key][x_val_key], car_state.kinematics_true[position_key][y_val_key], 0])

        if self.__use_speed:
            self.__speed = max(0, car_state.speed)
            state_speed = np.ones((59,255,1))
            state_speed.fill(self.__speed)
            print('speed:',self.__speed)
            # uint_img = np.array(state_speed*3315).astype('uint8')
            # grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
            # cv2.imshow('test',grayImage)
            state_buffer = np.concatenate([state_buffer, state_speed], axis=2)

        self.__total_reward = 0
        start_time = datetime.datetime.utcnow()
        end_time = start_time + datetime.timedelta(seconds=self.__max_epoch_runtime_sec)
        
        num_random = 0
        far_off = False

        while not done:
            collision_info = self.__car_client.getCollisionInfo()
            utc_now = datetime.datetime.utcnow()
            # added speed, so change little bit.
            # if (collision_info.has_collided or car_state.speed < 0.2 or utc_now > end_time or far_off):
            if (collision_info.has_collided or utc_now > end_time or far_off):
                print('end by collision: ',collision_info.has_collided)
                print('end by far off: ',far_off)
                print('Start time: {0}, end time: {1}'.format(start_time, utc_now), file=sys.stderr)
                print('Time elapsed: {0}'.format(utc_now-self.__the_start_time))
                print('Current best drive: {0}'.format(self.__best_drive))
                self.__car_controls.steering = 0
                self.__car_controls.throttle = 0
                self.__car_controls.brake = 1
                self.__car_client.setCarControls(self.__car_controls)
                
                time.sleep(4)
                if (utc_now > end_time):
                    print('timed out.')
                    print('Full autonomous run finished at {0}'.format(utc_now), file=sys.stderr)
                done = True
                self.__drive_distance = 0
                
                sys.stderr.flush()
            else:
                # The Agent should occasionally pick random action instead of best action
                do_greedy = np.random.random_sample()
                pre_state = copy.deepcopy(state_buffer)
                
                if (do_greedy < self.__epsilon or always_random):
                    num_random += 1
                    next_state, next_throttle = self.__model.get_random_state()
                    # next_throttle = tuple(np.random.rand(2)+1)
                    predicted_reward = 0
                    print('Do random {0}'.format(next_state), end='  ')
                else:
                    if self.__use_speed:
                        next_state, next_throttle, predicted_reward = self.__model.predict_state(pre_state,self.__use_speed)
                    else:
                        next_state, predicted_reward = self.__model.predict_state(pre_state, self.__use_speed)
                    print('Model say {0}'.format(next_state), end='  ')
                    # print('Model predicts {0}'.format(next_state), end='  ')

                # Convert the selected state to a control signal
                if self.__use_speed:
                    next_control_signals = self.__model.state_to_control_signals(next_state, car_throttle=next_throttle, use_speed=self.__use_speed)
                else:
                    next_control_signals = self.__model.state_to_control_signals(next_state, car_state=self.__car_client.getCarState())
                
                # Take the action
                self.__car_controls.steering = self.prev_steering + next_control_signals[0]
                # print('prev {0} -> changed {1}'.format(self.prev_steering, self.__car_controls.steering))
                if self.__car_controls.steering > 1.0:
                    self.__car_controls.steering = 1.0
                elif self.__car_controls.steering < -1.0:
                    self.__car_controls.steering = -1.0
                self.prev_steering = self.__car_controls.steering
                # print('normalized steering : ', self.prev_steering)
                to_print=np.round(next_control_signals[1:],2)
                print('next_control',"[{:.3f} {:.3f}]".format(to_print[0],to_print[1]), end='  ')
                
                
                self.__car_controls.throttle = next_control_signals[1]
                self.__car_controls.brake = next_control_signals[2]

                self.__car_client.setCarControls(self.__car_controls)
                
                # Wait for a short period of time to see outcome
                time.sleep(wait_delta_sec)

                # Observe outcome and compute reward from action
                state_buffer, state_lane = self.__get_image(use_lane=self.__use_lane)
                # (59,255,3) ()
                if self.__use_lane:
                    state_lane.shape = (59,255,1)
                    state_buffer = np.concatenate([state_buffer, state_lane],axis=2)

                if self.__use_handle:
                    angle = -int(self.prev_steering/0.05*4)
                    post_handle = self.__handles[angle].reshape(59,255,1)
                    cv2.imshow('handle',post_handle)
                    state_buffer = np.concatenate([state_buffer, post_handle],axis=2)

                if self.__use_speed:
                    self.__speed = max(0, self.__car_client.getCarState().speed)
                    state_speed = np.ones((59,255,1))
                    state_speed.fill(self.__speed)
                    # print('speed:',self.__speed)
                    # uint_img = np.array(state_speed*3315).astype('uint8')
                    # grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
                    # cv2.imshow('test',grayImage)
                    state_buffer = np.concatenate([state_buffer, state_speed], axis=2)

                post_state = state_buffer

                car_state = self.__car_client.getCarState()
                collision_info = self.__car_client.getCollisionInfo()
                reward, far_off = self.__compute_reward(collision_info, car_state)
                
                # Add the experience to the set of examples from this iteration
                pre_states.append(pre_state)
                post_states.append(post_state)
                rewards.append(reward)
                predicted_rewards.append(predicted_reward)
                actions.append(next_state)
                if self.__use_speed:
                    throttle.append(next_throttle)
                self.__total_reward = sum(rewards)

        # action수가 너무 적을경우, 그 회차의 학습을 진행하지 않음. #added 2021-03-09 by kang
        if len(actions) < 10:
            return self.__experiences, 0
            return self.__experiences, 0, 0

        is_not_terminal = [1 for i in range(0, len(actions)-1, 1)]
        is_not_terminal.append(0)
        self.__add_to_replay_memory('pre_states', pre_states)
        self.__add_to_replay_memory('post_states', post_states)
        self.__add_to_replay_memory('actions', actions)
        self.__add_to_replay_memory('throttles', throttle)
        self.__add_to_replay_memory('rewards', rewards)
        self.__add_to_replay_memory('predicted_rewards', predicted_rewards)
        self.__add_to_replay_memory('is_not_terminal', is_not_terminal)

        print('Percent random actions: {0}'.format(num_random / max(1, len(actions))))
        print('Epsilon: ',self.__epsilon)
        print('Num total actions: {0}'.format(len(actions)))
        
        if not always_random:
            self.__epsilon -= self.__per_iter_epsilon_reduction
            self.__epsilon = max(self.__epsilon, self.__min_epsilon)

        return self.__experiences, len(actions)
        # return self.__experiences, len(actions), utc_now - start_time

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
        if self.__use_speed:
            sampled_experiences['throttles'] = []
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
            if self.__use_speed:
                sampled_experiences['throttles'] += [experiences['throttles'][i] for i in idx_set]
            sampled_experiences['rewards'] += [experiences['rewards'][i] for i in idx_set]
            sampled_experiences['predicted_rewards'] += [experiences['predicted_rewards'][i] for i in idx_set]
            sampled_experiences['is_not_terminal'] += [experiences['is_not_terminal'][i] for i in idx_set]
            
        return sampled_experiences
        
    def __publish_batch_and_update_model(self, batches, batches_count, drive_time): # updateed 2021-03-09 by kang
        # Train and get the gradients
        print('Publishing epoch data and getting latest model from parameter server...')
        gradients = self.__model.get_gradient_update_from_batches(batches) 
    
        if (self.__num_batches_run > self.__batch_update_frequency + self.__last_checkpoint_batch_count):
            self.__model.update_critic()
            
            checkpoint = {}
            checkpoint['model'] = self.__model.to_packet(get_target=True)
            checkpoint['batch_count'] = batches_count
            checkpoint_str = json.dumps(checkpoint)
            # removed no more use of checkpoint dir
            # checkpoint_dir = os.path.join(os.path.join(self.__data_dir, 'checkpoint'), self.__experiment_name)

            # if not os.path.isdir(checkpoint_dir):
            #     try:
            #         os.makedirs(checkpoint_dir)
            #     except OSError as e:
            #         if e.errno != errno.EEXIST:
            #             raise
            #### saving evey experiences is waist of memory ### by Kang
            # # saving experiences by every update
            # tmp = os.path.join(os.path.join(checkpoint_dir, 'experiencedata'),'{0}.pkl'.format(self.__num_batches_run))
            # with open(tmp,'wb') as save_file:
            #     pkl.dump([self.__experiences, self.__epsilon, self.__best_drive], save_file)
            #     save_file.close()
            ####################################################
            # file_name = os.path.join(checkpoint_dir,'{0}.json'.format(self.__num_batches_run)) 
            # Removed cuz waist of memory. by Kang 21-03-11
            # with open(file_name, 'w') as f:
            #     print('Checkpointing to {0}'.format(file_name))
            #     f.write(checkpoint_str)

            self.__last_checkpoint_batch_count = self.__num_batches_run
            
            # 운행시간을 이용해서 가장 오래 걸린 시간을 best policy로 보고, best policy를 따로 저장. #added 2021-03-09 by kang
            # 만약 이번 회차의 운행시간이 가장 긴 운행시간일 경우에 best policy 저장
            # if self.__drive_distance*500 > self.__best_drive and self.__epsilon < 0.5:
            if self.__total_reward and self.__epsilon < 0.5:
                print("="*30)
                print("New Best Policy!!!!!!")
                print("="*30)
                self.__best_drive = self.__drive_distance*500
                bestpoint_dir = os.path.join(os.path.join(self.__data_dir, 'bestpoint'), self.__experiment_name)
                record_dir = os.path.join(os.path.join(self.__data_dir,'record'),self.__experiment_name)
                
                file_name = os.path.join(bestpoint_dir,'{0}.json'.format(self.__num_batches_run)) 
                record_file_name = os.path.join(record_dir,'{0}.txt'.format(self.__num_batches_run))
                # if dir is not exist, make directory by Kang 21-05-13
                if not os.path.isdir(bestpoint_dir):
                    try:
                        os.makedirs(bestpoint_dir)
                    except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise
                if not os.path.isdir(record_dir):
                    try:
                        os.makedirs(record_dir)
                    except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise
                # add {self.__num_batches_run}.json file to ./data/bestpoint/{self.__experiment_name}/
                with open(file_name, 'w') as f:
                    print('Add Best Policy to {0}'.format(file_name))
                    f.write(checkpoint_str)
                
                # saving bestpoint model and experiences to saved_point folder by Kang 21-03-12
                # saving {self.__num_batches_run}.json file to ./data/record/{self.__experiment_name}/
                with open(record_file_name, 'w') as f: #saving information of model by Seo 21-05-03
                    print('Add info to {0}'.format(record_file_name))
                    f.write(f'Total reward : {self.__total_reward}\n')
                    f.write(f'Start Time : {self.__the_start_time}\n')
                    f.write(f'Epoch Time : {datetime.datetime.utcnow()}\n')
                    f.write(f'Drive Distance : {self.__drive_distance}\n')

                self.__num_of_trial = 0

                # 처음부터 시작할때 최근의 상태를 알기 위하여 pickle로 experiences, epsilon 저장.   
                
                save_file = open(os.path.join(bestpoint_dir, f'{self.__num_batches_run}'+EXPERIENCE_FILENAME),'wb')
                pkl.dump([self.__experiences, self.__epsilon, self.__num_batches_run], save_file)
                save_file.close()

                self.__best_model = self.__model
                self.__best_experiences = self.__experiences
                self.__best_epsilon = self.__epsilon
            # 고민을 좀 해보자
            # elif self.__epsilon == 0.1:
            #     self.__num_of_trial += 1
            #     # print("="*30)
            #     # print("Reload best Model")
            #     # print("="*30)
            #     # self.__model = self.__best_model
            #     # self.__experiences = self.__best_experiences
            #     # self.__epsilon = self.__best_epsilon
            #     if self.__num_of_trial > 1000:
            #         print('='*16)
            #         print('increase epsilon')
            #         print('='*16)
            #         self.__epsilon = 0.5 # make epsilon reach to 0.5 by Kang 21-05-17
            #         self.__num_of_trial = 0
            

    def __compute_reward(self, collision_info, car_state):
        # Define some constant parameters for the reward function
        THRESH_DIST = 2.5                # The maximum distance from the center of the road to compute the reward function
        DISTANCE_DECAY_RATE = 1.2        # The rate at which the reward decays for the distance function

        # If the car has collided, the reward is always zero
        
        if (collision_info.has_collided):
            return 0.0, True
        
        # If the car is stopped, the reward is always zero
        speed = car_state.speed

        # to check about the speed, like upper change
        # if (speed < 0.2 or collision_info==True):
        if (collision_info==True):
            return 0.0, True
        
        #Get the car position
        position_key = bytes('position', encoding='utf8')
        x_val_key = bytes('x_val', encoding='utf8')
        y_val_key = bytes('y_val', encoding='utf8')

        car_point = np.array([car_state.kinematics_true[position_key][x_val_key], car_state.kinematics_true[position_key][y_val_key], 0])
        notmoved = np.round(self.__prev_car_point,2)==np.round(car_point,2)
        self.__prev_car_point = car_point
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
        
        # reward로 넣어줄때 m단위로 넣으면 너무 과적합될것 같아서 km단위로 넣기로 결정.
        # km단위로 설정시 너무 작아져, 500m단위로 결정.
        # 위에 방법을 취소하고, Decay_rate만큼을 곱한 distance를 뺀다!
        # 거리 = 시간 x 속도 (기본단위: m), 500을 나누면서 단위를 500m로 만듦.
        # 500m 단위로 설정시 너무 reward가 작아셔 음의 reward가 나오는 경우가 존재.
        # 따라서 100m단위로 수정했음
        # 몇일동안 해본 결과, 너무 느는 정도가 낮아서, 100m단위가 아닌 m단위를 쓰기로 결정-05_25 by Kang
        # 다시 500m단위로 수정
        self.__drive_distance += (0.05 * max(0, speed) )/500
        # 중앙까지 거리를 reward, 즉 양의 값으로 중앙가 가까울 경우 1, 멀경우 0에 가깝게 설정시
        # 차량의 속도를 더했을 경우 중앙에 도달하면 멈춰버리는 현상 발생.
        # 따라서 중앙에서의 거리를 penalty로 주어보기로 결정. 21-05-16 by Kang
        distance_reward = math.exp((distance * DISTANCE_DECAY_RATE))
        # 최대 distance_reward는 약 20정도를 갖기 때문에, min-max normalization.
        # 최소값은 e의 0제곱인 1, 최대값은 테스트 결과 약 20.6정도로 나왔기 때문에, min=1,max=21로
        # min-max 정규화
        distance_reward = (distance_reward-1)/(21-1)  # --> 멀면 1, 중앙이면 0에 가까움.
        if self.__use_speed:
            # reward를 0 이하로 내려갈 경우 0으로 했는데, 성과가 좋지 않아서 원래대로 음수도 나오도록 다시 설정한다.
            reward = -distance_reward*0.1 + self.__drive_distance*0.9
            # print(f'distance, drive {distance}m, {self.__drive_distance*100}m')
            # print(f'distance reward {distance_reward}')
            # print(f'drive_distance {self.__drive_distance}')
            # print('reward', reward)
            print('total reward {:.3f}'.format(np.round(self.__total_reward,3)),end='  ')
            print('drive distance {:.3f}(500m)'.format(np.round(self.__drive_distance,3)), end='  ')
            if notmoved[0] or notmoved[1]:
                print('stopped')
                return 0, distance > THRESH_DIST
            else:
                print()
                return reward, distance > THRESH_DIST
        else:
            return distance_reward, distance > THRESH_DIST

    def __get_image(self, use_lane=True):
        image_response = self.__car_client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
        image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
        
        image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
        img_4_to_3 = image_rgba[:,:,0:3]
        cv2.imshow('original',img_4_to_3)
        lane_detected=None
        if use_lane:
            lane_detected = road_lines(img_4_to_3, self.__lanes, self.__lane_model)
            cv2.imshow('original_lane', lane_detected)
            lane_detected = lane_detected[76:135,0:255,1].astype(float)*255
            cv2.imshow('cutted_lane',lane_detected)
        cv2.imshow('cutted',image_rgba[76:135,0:255,0:3])
        image_rgba = image_rgba[76:135,0:255,0:3].astype(float)
        image_rgba = image_rgba.reshape(59, 255, 3)
        

        cv2.waitKey(1)
        return image_rgba, lane_detected

    def __init_road_points(self):
        self.__road_points = []
        car_start_coords = [12961.722656, 6660.329102, 0]
        road = ''
        if not random_respawn:
            road = 'road_lines.txt'
        else:
            road = 'origin_road_lines.txt'
        with open(os.path.join(os.path.join(self.__data_dir, 'data'), road), 'r') as f:
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
        
        # added return to origin by Kang 21-03-10
        if not random_respawn:
            if random_line_index==0:
                random_interp = 0.9
            else:
                random_interp = 0.15    # changed by GY 21-03-10
            # Pick a random direction to face
            random_direction_interp = 0.4 # changed by GY 21-03-10
        else:
            random_interp = (np.random.random_sample() * 0.4) + 0.3 
            random_direction_interp = np.random.random_sample()

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
                        -20 : cv2.cvtColor(cv2.imread(self.__handle_dir+'right20.png'), cv2.COLOR_BGR2GRAY),
                        -40 : cv2.cvtColor(cv2.imread(self.__handle_dir+'right40.png'), cv2.COLOR_BGR2GRAY),
                        -60 : cv2.cvtColor(cv2.imread(self.__handle_dir+'right60.png'), cv2.COLOR_BGR2GRAY),
                        -80 : cv2.cvtColor(cv2.imread(self.__handle_dir+'right80.png'), cv2.COLOR_BGR2GRAY),
                        20 : cv2.cvtColor(cv2.imread(self.__handle_dir+'left20.png'), cv2.COLOR_BGR2GRAY),
                        40 : cv2.cvtColor(cv2.imread(self.__handle_dir+'left40.png'), cv2.COLOR_BGR2GRAY),
                        60 : cv2.cvtColor(cv2.imread(self.__handle_dir+'left60.png'), cv2.COLOR_BGR2GRAY),
                        80 : cv2.cvtColor(cv2.imread(self.__handle_dir+'left80.png'), cv2.COLOR_BGR2GRAY)}

print('If you want to load best policy, Enter "y" ,otherwise "n"')
# MODEL_FILENAME = 'best_model.json' if input()=='y' else None
print('If you want to use handle, Enter "y", otherwise "n"')
handle_choose = True
if input() == 'y': pass
else: handle_choose = False
print('If you want to use lane detection, Enter "y", otherwise "n"')
lane_choose = True
if input() == 'y': pass
else: lane_choose = False
if lane_choose:
    trained_model  = load_model('full_CNN_model.h5')
print('If you want to do speed handling, Enter "y", otherwise "n"')
speed_choose = True
if input() == 'y': pass
else: speed_choose = False
agent = DistributedAgent(use_handle=handle_choose, use_lane = lane_choose, use_speed = speed_choose)
agent.start()