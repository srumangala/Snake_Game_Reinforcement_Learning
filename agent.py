import torch
import random
import numpy as np
from collections import deque # data structure to store memory
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot 

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 # Learning rate

class Agent:

    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # to control the randomness
        self.gamma = 0.9 # discount rate (Smaller than 1)
        self.memory = deque(maxlen = MAX_MEMORY) # popleft() - remove elements from the left
        self.model = Linear_QNet(11,256,3) # Input size, Hidden Layer size, Output size
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)
        # TODO: model, trainer

    def get_state(self, game):
        head = game.snake[0] # the head of the snake

        # Points next to the head in all directions
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x , head.y - BLOCK_SIZE)
        point_d = Point(head.x , head.y + BLOCK_SIZE)
        
        # The current direction. Only one of these will be one
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight. If there is a collision in the next point in the same direction
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger Right. If there is a collision in the next point when turning right
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # Danger Left. If there is a collision in the next point when turning left
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            # Move direction. Direction of movement. Only one of these is 1 at a state
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int) # bool values are converted to int

    def remember(self, state, action, reward, next_state, done): # done = game over
        self.memory.append((state, action, reward, next_state, done)) # if exceeds MAX_MEMORY, popleft()

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples 
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves : tradeoff exploration / exploitation
        # For more number of games, the epsilon will get smaller
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon: # the smaller the epsilon, the randomness of the move will become lesser. For a negative epsilon, it becomes non-random
            move = random.randint(0, 2) # index for the final move
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float) # converting the state to a tensor
            prediction = self.model(state0) # predict the action based on the state tensor
            move = torch.argmax(prediction).item() # the index of the argument with max value ( the position)
            final_move[move] = 1
        
        return final_move


def train(): # Global function
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0 # highest score ever
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory 
        agent.train_short_memory(state=state_old, action=final_move,reward=reward,next_state=state_new,done=done)

        # remember
        agent.remember(state=state_old, action=final_move,reward=reward,next_state=state_new,done=done)

        if done:
            # train long memory (trains on all previous moves), plot the result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()
