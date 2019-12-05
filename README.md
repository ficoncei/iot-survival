# iot-survival

# What is iot-survival?
iot-survival is a problem conception (a stochastic control optimisation problem) and solution that takes an IoT device energetically (under) fed by an energy harvester and makes optimal security decisions (actions) to ensure both energetic survival and assuring packet protection. 

# Machine Learning for IoT device survival
An online Deep Reinforcement Learning approach (Deep Q-learning) is used where a neural network approximates the action value function and is updated as the device's experience progresses in time. Security decisions are made that result in an adjustment of the security overhead transmitted over the air interface to the available energy in the device's battery. This allows for device's survival, available battery energy increase and data availabiltity gains.

# Neural network approach
The neural network has:
- 2 input neurons to account for the current battery state and security level in use
- 4 hidden layers with 6 neurons each (no under fitting, no over fitting)
- 7 output neurons corresponding to the action selection

# Training
The training is done online by using the collected reward from each action selection as an input to the Mean Squared Error function , to compare with the current action preference given by the current neural network weight set.

# Outputs
The script outputs a comparison plot between the Deep Q-learning approach and the current state of the art that always tries to enforce maximum security.
