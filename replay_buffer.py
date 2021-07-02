from collections import deque
import random

class ReplayBuffer():
    def __init__(self,capacity=800):
        self.buffer = deque(maxlen=capacity)
    
    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
    
    def sample(self,batch_size=32):
        state,action,reward,next_state,done =  zip(*random.sample(self.buffer,batch_size))
        return((state,action,reward,next_state,done))