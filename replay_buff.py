import random

class ReplayMemory(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        
    def push(self, state, action, reward, next_state, coords,next_coords,done):
        experience = (state, action, reward, next_state, coords,next_coords,done)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        coords_batch = []
        done_batch = []
        next_cords_batch = []
        batch = random.sample(self.buffer, batch_size)
        
        for experience in batch:
            state, action, reward, next_state, coords,next_coords,done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            coords_batch.append(coords)
            next_cords_batch.append(next_coords)
            done_batch.append(done)
        
        return (state_batch, action_batch, reward_batch, next_state_batch, coords_batch,next_cords_batch,done_batch)
    
    def truncate(self):
        self.buffer = self.buffer[-self.max_size:]
    
    def __len__(self):
        return len(self.buffer)