import time
import cv2 as cv
import numpy as np
import gamegrid as env
from queue import deque
import tensorflow as tf
from random import randint
from memory import Memory
from DQNetwork import DQNetwork
import statistics as stats

def create_environment():
    """creates a gamegrid environment"""

    game = env.GameGrid(20)

    up = [1, 0, 0, 0]
    down = [0, 1, 0, 0]
    left = [0, 0, 1, 0]
    right = [0, 0, 0, 1]


    possible_actions = [up, down, left, right]
    return game, possible_actions
#print(environment.preprocess_frame())

# cv.imshow('img',environment.frame)
# cv.waitKey(0)
# print(environment.perform_action(2))
# cv.imshow('img',environment.frame)
# cv.waitKey(0)

game, possible_actions = create_environment()

stack_size = 4 # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

def  stack_frames(stacked_frames, state, is_new_episode):
    """stacks the frames"""

    # Preprocess frame
    frame = game.preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames

# setting hyperparameters for training the model
# Model Hyperparameters
state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = 4              # 4 possible actions: up, down, left, right
learning_rate =  0.002      # learning rate

# Training Hyperparameters
total_episodes = 2000        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 64

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95               # Discounting rate

# Memory Hyperparameters
pretrain_length = 5000   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 25000          # Number of experiences the Memory can keep

# Flag for if training needs to be done
training = False


tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

memory = Memory(max_size = memory_size)

# Reset the environment
game.reset()

for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        # get a state
        state, color_frame = game.createImage()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    # Random action
    action = possible_actions[randint(0, 3)]

    # Get rewards
    terminal, reward = game.perform_action(action)

    # Look if the episode is finished
    # done = game.is_episode_finished()

    # If episode ends
    if terminal:

        # episode finishes
        next_state = np.zeros(state.shape)

        # Add experience to memory
        memory.add((state, action, reward, next_state, terminal))

        # Start a new episode
        game.reset()

        # get a state
        state, color_frame = game.createImage()

        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        # Get next state
        next_state, color_frame = game.createImage()
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        # Add experience to memory
        memory.add((state, action, reward, next_state, terminal))

        # Our state is now the next_state
        state = next_state

#To launch tensorboard : tensorboard --logdir=/tensorboard/dqn/1
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/dqn/1")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()



def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    """With probablity ϵ select a random action, otherwise selectbest action - argmaxQ(state, action)"""

    # Epsilon greedy strategy
    # Choose action a from state s using epsilon greedy.

    # First randomize a number
    exp_exp_tradeoff = np.random.rand()

    # improved version of epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = actions[randint(0, 3)]

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = actions[int(choice)]

    return action, explore_probability



def display(start, action_stack):
    """display series of frames of player taking actions"""

    game, possible_actions = create_environment()
    game.reset()
    game.setup_grid()
    game.setStartLoc(start)
    game.update_player()

    gray_frame, color_frame = game.createImage()

    cv.imshow('Maze', cv.resize(color_frame,(500,500)))
    cv.waitKey(50)

    for action in action_stack:
        game.perform_action(action)
        gray_frame,color_frame = game.createImage()
        cv.imshow('Maze', cv.resize(color_frame,(500,500)))
        cv.waitKey(50)


# save model
saver = tf.train.Saver()


if training == True:
    with tf.Session() as sess:

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        # Init the game
        game.reset()

        for episode in range(total_episodes):
            # Set step to 0
            step = 0

            # Statistics
            rewards_statistics = []
            loss_statistics = []
            episode_statistics = []

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state
            game.reset()
            state, color_frame = game.createImage()

            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                step += 1

                # Increase decay_step
                decay_step +=1

                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

                # Do the action
                terminal, reward = game.perform_action(action)


                # Add the reward to total reward
                episode_rewards.append(reward)
                #print(episode_rewards)

                # If the game is finished
                if terminal:
                    # the episode ends so no next state
                    next_state = np.zeros((84,84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))

                    memory.add((state, action, reward, next_state, terminal))

                else:
                    # Get the next state
                    next_state, color_frame = game.createImage()

                    # Stack frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)


                    # Add experience to memory
                    memory.add((state, action, reward, next_state, terminal))

                    # st+1 is now our current state
                    state = next_state

                # learning
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # When in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)


                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                    feed_dict={DQNetwork.inputs_: states_mb,
                                               DQNetwork.target_Q: targets_mb,
                                               DQNetwork.actions_: actions_mb})

#                 print('Episode: {}'.format(episode),
#                               'Total reward: {}'.format(total_reward),
#                               'Training loss: {:.4f}'.format(loss),
#                               'Explore P: {:.4f}'.format(explore_probability))

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()


            if (episode%25==0):

                # print('In Episode '+str(episode))
                episode_statistics.append(episode)

                total_reward = np.sum(episode_rewards)
                # print('Reward here is '+ str(total_reward))
                rewards_statistics.append(total_reward)

                # print('Loss is '+str(loss))
                loss_statistics.append(loss)


            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")
    
    
    print('Training Statistics')
    stats.showLossStatistics(episode_statistics, loss_statistics)
    stats.showRewardStatistics(episode_statistics, rewards_statistics)
    
    
else:

    with tf.Session() as sess:

        game, possible_actions = create_environment()
        success= 0
        # Load the model
        saver.restore(sess, "./models/model.ckpt")
        game.reset
        for i in range(50):

            action_stack = []
            totalScore = 0
            step = 0

            terminal = False

            game.reset()

            state, color_frame = game.createImage()

            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while not terminal and step < 500:
                # Take the biggest Q value (= the best action)
                Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

                # Take the biggest Q value (= the best action)
                choice = np.argmax(Qs)
                action = possible_actions[int(choice)]

                terminal, reward = game.perform_action(action)

                #print(str(action) + str(terminal) + str(reward))

                action_stack.append(action)
                totalScore += reward
                step+=1

                if terminal:
                    success +=1
                    cv.destroyAllWindows()
                    display(game.getStartLoc(), action_stack)
                    cv.destroyAllWindows()
                    break

                else:
                    # print("Not reached goal " + str(totalScore))
                    next_state, color_frame = game.createImage()
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    state = next_state
                    #cv.imshow('img', cv.resize(stacked_frames[3],(500,500)))
                    #time.sleep(5)
                    #cv.waitKey(50)

            score = totalScore
            print("Score: ", score)
            print("Terminal: ", success, " Rate :", success/50.0)

            

        del game
