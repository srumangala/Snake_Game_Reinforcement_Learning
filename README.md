# Deep Q Learning : Snake Game

This is a project I did to get into Deep Q Networks and Reinforcement Learning.

Source : https://www.youtube.com/watch?v=L8ypSXwyBds

While I took most of the code from the source, I only did so after properly understanding the concepts.

## Goal

The goal of the project is to train an agent to learn to play the snake game using Deep Q Networks. The snake is rewarded everytime it consumes food and penalized everytime it hits the boundaries or itself. The game is developed using pygame and the UI looks like a traditional snake game. Everytime the snake consumes food, it also grows one block in size.

## 📁 Project Structure

| File | Description |
|------|-------------|
| `game.py` | Snake environment using Pygame |
| `model.py` | Neural network model (DQN) |
| `agent.py` | RL agent: action selection, replay buffer, training loop |
| `helper.py` | Plotting utilities for training progress |
| `snake_game.py` | Human-playable Snake |

---

## Steps to run

Just run agent.py after creating a virtual environment and installing dependencies. It appears that the model learns during 70-80th time of playing

## The Agent and Model Structure:

### States

The state consists of 11 values. They are:

[
    danger straight, danger right, danger left,
    direction left, direction right, direction up, direction down,
    food left, food right, food up, food down
]

**Danger indicators**  
Binary values showing whether moving straight, right, or left would cause a collision (wall or body). The danger elements specify if there is a danger of collision when the snake either proceeds forward, takes a right or left turn. A collision happens when the snake hits the boundaries of the wall or itself.

**Movement direction**  
The direction values specify the current direction of movement of the snake. Only one among the 4 can be true at a time.

**Food position**  
Binary values showing the relative location of food with respect to the snake's head.  
Multiple can be true (e.g., food is up-left).

### Actions

The snake can have 3 actions. These are one hot encoded.

[1, 0, 0] -> Straight
[0, 1, 0] -> Right Turn
[0, 0, 1] -> Left Turn

###  Model Architecture

A simple feed-forward neural network:

- **Input:** 11 features  
- **Hidden layer:** 256 units, ReLU activation  
- **Output:** 3 Q-values (one per action)  
- **Loss:** MSE  
- **Optimizer:** Adam  

---

## The workflow

The agent uses two forms of training:

### **1. Short Training**
Performed after every move.  
A single transition `(state, action, reward, next_state)` is used for a one-step TD update.

### **2. Long Training**
Performed after each game ends.  
A minibatch (e.g., 1000 transitions) is sampled from the replay memory and trained using the DQN target.


### Full loop

1. Obtain current state
2. Predict the next action. The action is performed randomly in the beginning and as the number of games increases, the Neural Network predicts the action. [Exploration vs Exploitation]
3. Play the move - This results in reward
4. Obtain next state
5. If game is not over, start short training based on the current state, action, reward, next state
6. Calculate the new Q value using Bellman equation
7. Update the weights using backpropagation
8. Save the state, action, next state, reward and done status to the memory
9. Repeat the same until game over.
10. When game is over, perform the long training (Q values are updated here also)
11. Update weights using backpropagation


## Important Pre requisites 
1. Neural Networks
2. Bellman Equations
3. Q Learning
