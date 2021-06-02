import time
from keras import regularizers
import numpy as np
import json
import threading
import os,sys
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, clone_model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam, Adagrad, Adadelta
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
from keras.preprocessing import image
from keras.initializers import random_normal
from tensorflow.python.keras import backend as K
# Prevent TensorFlow from allocating the entire GPU at the start of the program.
# Otherwise, AirSim will sometimes refuse to launch, as it will be unable to 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)
np.set_printoptions(threshold=sys.maxsize)  
# A wrapper class for the DQN model
class RlModel():
    def __init__(self, weights_path, train_conv_layers, run = True, use_handle = True, use_lane = True, use_speed = True):
        #self.__angle_values = [-1, -0.5, 0, 0.5, 1]
        self.__angle_values = [-0.5, -0.25, 0, 0.25, 0.5] #continuous state
        self.__throttle_values = [[1,0],[0,1]]
        print('handle, lane, speed', use_handle,use_lane,use_speed)

        self.__nb_actions = 5
        self.__nb_speed_actions = 2
        self.__gamma = 0.99
        self.__use_handle = use_handle
        self.__use_lane = use_lane
        self.__use_speed = use_speed

        #Define the model
        activation = 'relu'

        option = np.array([use_handle,use_lane,use_speed])
        depth_check = np.array([True,True,True])
        depth = 3 + len(depth_check[option])
        self.__depth = depth
        pic_input = Input(shape=(59,255,self.__depth))
        # if self.__use_handle and self.__use_lane and self.__use_speed:
        #     pic_input = Input(shape=(59,255,6)) #with handle and with lane and with handle and with speed
        # elif (self.__use_handle and self.__use_lane) or (self.__use_lane and self.__use_speed) or (self.__use_handle and self.__use_speed):
        #     pic_input = Input(shape=(59,255,5)) #with handle and with lane or with handle and with speed
        # elif self.__use_handle or self.__use_lane or self.__use_speed:
        #     pic_input = Input(shape=(59,255,4)) #only with handle or lane
        # else:
        #     pic_input = Input(shape=(59,255,3)) #without handle and lane
        
        
        img_stack = Conv2D(32, (3, 3), activation=activation, name='convolution0', padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(1e-4))(pic_input)
        img_stack = MaxPooling2D(pool_size=(2,2))(img_stack)
        img_stack = Conv2D(64, (3, 3), activation=activation, name='convolution1', padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(1e-4))(img_stack)
        img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
        img_stack = Conv2D(128, (3, 3), activation=activation, name='convolution2', padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(1e-4))(img_stack)
        img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
        img_stack = Flatten()(img_stack)
        if not run:
            img_stack = Dropout(0.2)(img_stack)

        img_stack = Dense(128, name='rl_dense1', kernel_initializer=random_normal(stddev=0.01))(img_stack)

        #with handle
        img_stack = BatchNormalization()(img_stack)
        img_stack = Dense(128, name='rl_dense2', kernel_initializer=random_normal(stddev=0.01))(img_stack)
        img_stack = BatchNormalization()(img_stack)
        img_stack = Dense(128, name='rl_dense3', kernel_initializer=random_normal(stddev=0.01))(img_stack)
        img_stack = BatchNormalization()(img_stack)
        
        if self.__use_speed:
            output_throttle = Dense(self.__nb_speed_actions, name='rl_throttle_output', kernel_initializer=random_normal(stddev=0.01), activation='sigmoid')(img_stack)
        output = Dense(self.__nb_actions, name='rl_output', kernel_initializer=random_normal(stddev=0.01))(img_stack)

        opt = Adam()
        if self.__use_speed:
            self.__action_model = Model(inputs=[pic_input], outputs=[output, output_throttle])
            losses = {
                "rl_output": "mean_squared_error",
                "rl_throttle_output": "mean_squared_error",
            }
            lossWeights = {"rl_output":2.0, "rl_throttle_output": 1.0}
        else:
            self.__action_model = Model(inputs=[pic_input], outputs=output)
            losses = {
                "rl_output": "mean_squared_error"
            }
            lossWeights = {"rl_output":1.0}
        
        self.__action_model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights)
        self.__action_model.summary()

        keras.utils.plot_model(self.__action_model, 'Model.png', show_shapes=True)
        
        # If we are using pretrained weights for the conv layers, load them and verify the first layer.
        if (weights_path is not None and len(weights_path) > 0):
            print('Loading weights from my_model_weights.h5...')
            print('Current working dir is {0}'.format(os.getcwd()))
            self.__action_model.load_weights(weights_path, by_name=True)
            
            print('First layer: ')
            w = np.array(self.__action_model.get_weights()[0])
            print(w)
        else:
            print('Not loading weights')

        # Set up the target model. 
        # This is a trick that will allow the model to converge more rapidly.
        self.__action_context = tf.get_default_graph()
        self.__target_model = clone_model(self.__action_model)

        self.__target_context = tf.get_default_graph()
        self.__model_lock = threading.Lock()

    def get_last_conv_layer(self, input_image):
        last_conv_layer = self.__action_model.get_layer('convolution2')
        grad_model = keras.models.Model()

    # A helper function to read in the model from a JSON packet.
    # This is used both to read the file from disk and from a network packet
    def from_packet(self, packet):
        with self.__action_context.as_default():
            self.__action_model.set_weights([np.array(w) for w in packet['action_model']])
            self.__action_context = tf.get_default_graph()
        if 'target_model' in packet:
            with self.__target_context.as_default():
                self.__target_model.set_weights([np.array(w) for w in packet['target_model']])
                self.__target_context = tf.get_default_graph()

    # A helper function to write the model to a JSON packet.
    # This is used to send the model across the network from the trainer to the agent
    def to_packet(self, get_target = True):
        packet = {}
        with self.__action_context.as_default():
            packet['action_model'] = [w.tolist() for w in self.__action_model.get_weights()]
            self.__action_context = tf.get_default_graph()
        if get_target:
            with self.__target_context.as_default():
                packet['target_model'] = [w.tolist() for w in self.__target_model.get_weights()]

        return packet

    # Updates the model with the supplied gradients
    # This is used by the trainer to accept a training iteration update from the agent
    def update_with_gradient(self, gradients, should_update_critic):
        with self.__action_context.as_default():
            action_weights = self.__action_model.get_weights()
            if (len(action_weights) != len(gradients)):
                raise ValueError('len of action_weights is {0}, but len gradients is {1}'.format(len(action_weights), len(gradients)))
            
            print('UDPATE GRADIENT DEBUG START')
            
            dx = 0
            for i in range(0, len(action_weights), 1):
                action_weights[i] += gradients[i]
                dx += np.sum(np.sum(np.abs(gradients[i])))
            print('Moved weights {0}'.format(dx))
            self.__action_model.set_weights(action_weights)
            self.__action_context = tf.get_default_graph()

            if (should_update_critic):
                with self.__target_context.as_default():
                    print('Updating critic')
                    self.__target_model.set_weights([np.array(w, copy=True) for w in action_weights])
            
            print('UPDATE GRADIENT DEBUG END')
            
    def update_critic(self):
        with self.__target_context.as_default():
            self.__target_model.set_weights([np.array(w, copy=True) for w in self.__action_model.get_weights()])
    
            
    # Given a set of training data, trains the model and determine the gradients.
    # The agent will use this to compute the model updates to send to the trainer
    def get_gradient_update_from_batches(self, batches):
        pre_states = np.array(batches['pre_states'])
        post_states = np.array(batches['post_states'])
        rewards = np.array(batches['rewards'])
        actions = list(batches['actions'])
        if self.__use_speed:
            throttles = list(batches['throttles'])
        is_not_terminal = np.array(batches['is_not_terminal'])
        
        with self.__action_context.as_default():
            if self.__use_speed:
                labels_1,labels_2 = self.__action_model.predict([pre_states], batch_size=32)
            else:
                labels = self.__action_model.predict([pre_states], batch_size=32)
        
        # Find out what the target model will predict for each post-decision state.
        with self.__target_context.as_default():
            if self.__use_speed:
                q_futures_1, q_futures_2 = self.__action_model.predict([pre_states], batch_size=32)
            else:
                q_futures = self.__target_model.predict([post_states], batch_size=32)
        
        # Apply the Bellman equation
        if self.__use_speed:
            q_futures_1_max = np.max(q_futures_1, axis=1)
            q_futures_2_max = np.max(q_futures_2, axis=1)
            q_labels_1 = (q_futures_1_max * is_not_terminal * self.__gamma) + rewards
            q_labels_2 = (q_futures_2_max * is_not_terminal * self.__gamma) + rewards

        else:
            q_futures_max = np.max(q_futures, axis=1)
            q_labels = (q_futures_max * is_not_terminal * self.__gamma) + rewards
        
        # Update the label only for the actions that were actually taken.
        if self.__use_speed:

            for i in range(0, len(actions), 1):
                labels_1[i][actions[i]] = q_labels_1[i]
                labels_2[i][throttles[i]] = q_labels_2[i]
        else:

            for i in range(0, len(actions), 1):
                labels[i][actions[i]] = q_labels[i]

        # Perform a training iteration.
        with self.__action_context.as_default():
            original_weights = [np.array(w, copy=True) for w in self.__action_model.get_weights()]
            if self.__use_speed:
                self.__action_model.fit([pre_states], [labels_1, labels_2], epochs=1, batch_size=32, verbose=1)
            else:
                self.__action_model.fit([pre_states], labels, epochs=1, batch_size=32, verbose=1)
            
            # Compute the gradients
            new_weights = self.__action_model.get_weights()
            gradients = []
            dx = 0
            for i in range(0, len(original_weights), 1):
                gradients.append(new_weights[i] - original_weights[i])
                dx += np.sum(np.sum(np.abs(new_weights[i]-original_weights[i])))
            print('change in weights from training iteration: {0}'.format(dx))
        
        print('END GET GRADIENT UPDATE DEBUG')

        # Numpy arrays are not JSON serializable by default
        #return [w.tolist() for w in gradients]

    # Performs a state prediction given the model input
    # def predict_state(self, observation):
    def predict_state(self, observation, use_speed):
        # Our model only predicts on a single state.
        # Take the latest image
        observation = observation.reshape(1,59,255,self.__depth)            

        with self.__action_context.as_default():
            predicted_qs = self.__action_model.predict([observation])
        # print('predicted_qs',predicted_qs)
        if use_speed:
            # print('sum of this', sum(sum(predicted_qs[0])))
            # print('speed sigmoid', predicted_qs[1])
            # Select the action with the highest Q value
            predicted_state = np.argmax(predicted_qs[0])
            predicted_throttle = np.argmax(predicted_qs[1])
            # return (predicted_state, predicted_qs[0][0][predicted_state])
            return (predicted_state, predicted_throttle, predicted_qs[0][0][predicted_state])
        else:
            print('sum of this', sum(sum(predicted_qs)))
            # Select the action with the highest Q value
            predicted_state = np.argmax(predicted_qs)
            return (predicted_state, predicted_qs[0][predicted_state])
    
    # Convert the current state to control signals to drive the car.
    # As we are only predicting steering angle, we will use a simple controller to keep the car at a constant speed
    def state_to_control_signals(self, state, car_state=None, car_throttle=None, use_speed=False):
        # (angle, speed up, break) marked by kang 21-03-12
        if use_speed:
            return (self.__angle_values[state], self.__throttle_values[car_throttle][0],self.__throttle_values[car_throttle][1])
            throttle=car_throttle[0].item()
            car_break = car_throttle[1].item()
            if throttle==car_break and throttle==0:
                throttle=1
            # return (self.__angle_values[state], throttle, car_break)
            if throttle>=car_break:
                return (self.__angle_values[state], throttle, 0)
            else:
                return (self.__angle_values[state], 0, car_break)
        else:
            if car_state.speed > 7:
                return (self.__angle_values[state], 0, 1)
            else:
                return (self.__angle_values[state], 1, 0)
            
    
    # Gets a random state
    # Used during annealing
    def get_random_state(self):
        return np.random.choice(5, 1)[0], np.random.choice(2,1)[0]
        #return np.random.randint(low=0, high=(self.__nb_actions) - 1)
