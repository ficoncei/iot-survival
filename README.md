# IoT-survival

# What is IoT-survival?
iot-survival is a problem conception (a stochastic control optimisation problem) and solution that takes an IoT device energetically underfed by an energy harvester and makes optimal security decisions on security levels to use (actions) to ensure both energetic survival and assuring packet protection. Different levels will add different security overheads to packets being transmitted (see systemmodel.png).

# Machine Learning for IoT device survival
A Deep Reinforcement Learning approach is used where an Actor-Critic method trains a double headed neural network (see Actor-Critic.png). The neural network approximates the action value function and is updated as the device's experience progresses in time. Security decisions are made that result in an adjustment of the security overhead transmitted over the air interface to the available energy in the device's battery. This allows for device's survival, available battery energy increase and data availabiltity gains.

# Neural network approach
The neural network has:
- 2 input neurons to account for the current battery state and security level in use
- 4 hidden layers with 6 neurons each
- 2 output layers with 7 output neurons each, corresponding to action selection preferences (see Actor-Critic.png)

# Training
The training is done online by using the collected reward from each action selection as an input to the Mean Squared Error function , to compare with the current action preference given by the current neural network weight set.

# Outputs
The script outputs a comparison plot between the Deep Learning actor-Critic approach and the current state of the art that always tries to enforce maximum security, in terms of the total collected rewards while in the process of learning. Resulting plot shows an IoT device using a stochastic optimal policy thriving and continuosly collecting high cumulative rewards and a state of the art that results in battery depletion.

