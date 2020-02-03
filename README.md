# maze-deep-reinforcement-learning

Deep Reinforcement Q Learning (Maze) involves a player at a random starting location on a 20 X 20 environment (grid) trying to find the treasure at a fixed location on the grid.


## Environment

There are multiple obstacles on the grid that the player cannot enter such as walls. However, the player can enter pits with mixed results because of the nature of the pits.

Pits can be of 2 kinds:

* *Shallow pit* - player can immediately escape with some negative reward
* *Deep pit*    - player cannot escape and the hunt for treasure is over

The following are details about the environment and rewards for various elements:

|Element|Initialization|Number|Reward|
|:---:|:---:|:---:|:---:|
|Player|Random|1|-|
|Wall|Fixed|100|-0.2|
|Shallow Pit|Fixed|4|-0.3|
|Deep Pit|Fixed|4|-100|
|Treasure|Fixed|1|+100|


### Maze Environment Start State

|Element|Grid Color|
|:---:|:---:|
|Treasure|Green|
|Wall|Navy Blue|
|Shallow Pit|Yellow|
|Deep Pit|Red|

![alt text](https://github.com/prakashdontaraju/maze-deep-reinforcement-learning/blob/master/maze-deep-q-learning/maze-grid.PNG)



## Objective

Train the player to choose actions by utilizing a Neural Network to predict Q-values for each state so as to find the treasure within the grid

By informing the fixed locations of treasure, walls and pits as inputs, the neural network will be trained to estimate the Q-values of all possible states.

Learn the optimal policy to find the treasure on the grid with reasonable accuracy from a randomized location


## Process

* Actions
  - 4 actions (Up, Down, Left, Right) are represented with a 4 X 1 vector
  - The position of 1 in the vector indicates the direction in which the player jumps (up, down, left and right)
  - Actions are one-hot encoded.

* Grid to frames (images)
  - Grid is converted to 200X200 size frame which includes player position along with all other elements

* Pre-processing
  - Frame is resized to 84X84 and converted to grayscale

* Normalization
  - Image is normalized and sent to the neural network as input

* Stacking
  - Stack of 4 consecutive frames are sent to the neural network as input at each step to exploit temporal information
  - Player performs action based on target q-value

* Hyperparameters
  - Hyperparameters are set and the DQ network is initialized

* Memory
  - Memory is created to store experiences from random actions for the player to not favor certain actions during training
  - Experiences are added to memory during training
  - Batches of 64 are sampled from memory to train the neural network

* Training the model
  - Model is trained for a specified number of episodes
  - Episode reaches terminal state when the player either finds treasure or falls into a deep pit
  - Episode is complete when the player either reaches terminal state or performs 100 actions

* Testing the model
  - Model is tested multiple times while recording reward and loss statistics to plot graphs
  - Frames (images) from start to end are displayed when player reaches terminal state in an episode


## Prerequisites

What are the tools you need to install?

```
You must have administrator access to install the following:

TensorFlow        TensorFlow 1.12
Python            3.7.3 or newer
CUDA              CUDA 10.0
cuDNN             cuDNN 7.6.0
Python Libraries  OpenCV-Python, numpy, matplotlib
Text Editor       VS Code or any other
```


## Deployment

By following the prerequisites and process, you'll be able to deploy our project, train the neural network and find the treasure consistently.


## Authors

* **Prakash Dontaraju** [LinkedIn](https://www.linkedin.com/in/prakashdontaraju) [Medium](https://medium.com/@wittygrit)
* **Nikhil G** [LinkedIn](https://www.linkedin.com/in/nikhil-g-95861bb7)
