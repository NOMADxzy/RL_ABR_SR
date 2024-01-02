import torch
import numpy as np
import linearNet.buffer as buffer
import linearNet.train as train

ABR_STATE_DIM = 7
MAX_BUFFER = 1000000
PLACE_STATE_DIM = 3
ABR_DIM = 6
SPACE_DIM = 2
A_MAX = 1
class Agent:
    def __init__(self):

        ram1 = buffer.MemoryBuffer(MAX_BUFFER)
        ram2 = buffer.MemoryBuffer(MAX_BUFFER)
        abr_actior = train.Trainer(ABR_STATE_DIM, ABR_DIM, A_MAX, ram1)
        place_actor = train.Trainer(PLACE_STATE_DIM, SPACE_DIM, A_MAX, ram2)

        self.abr_actor = abr_actior
        self.place_actor = place_actor
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def load(self, abr_epoch=0, place_epoch=0):
        if abr_epoch > 0:
            self.abr_actor.load_models(msg="abr", episode=abr_epoch)
        if place_epoch > 0:
            self.place_actor.load_models(msg="place", episode=place_epoch)

    def get_exploration_action(self, state):
        abr_state = np.zeros((ABR_STATE_DIM,))
        for i in range(0,4):
            abr_state[i] = state[2][-(i+1)] # 历史吞吐量
        abr_state[4] = state[3][-1] # 下载时间
        abr_state[5] = state[1][-1] # 缓冲区大小
        abr_state[6] = state[0][-1] # 上个动作
        place_state = np.asarray([abr_state[0], state[6][-1], state[7][-1]])
        _, bit_rate, _ = self.abr_actor.get_exploration_action(np.float32(abr_state))
        _, place_choice, _ = self.place_actor.get_exploration_action(np.float32(place_state))
        return bit_rate,place_choice