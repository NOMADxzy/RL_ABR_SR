import numpy as np
import random

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
# TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 40.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = "./video_size/ori/video_size_"
sigma = 4
K_RANGE = [5, 10]
MAX_K_FACTOR = 10


def initialize_tasks(task_list, all_file_names):
    task2idx = {}
    for task_id in task_list:
        for trace_id in range(len(all_file_names)):
            if task_id in all_file_names[trace_id]:
                try:
                    task2idx[task_id].append(trace_id)
                except:
                    task2idx[task_id] = []
                    task2idx[task_id].append(trace_id)
    # assert(len(task2idx)==len(task_list))
    return task2idx


class Environment:
    def __init__(
        self,
        all_cooked_time,
        all_cooked_bw,
        random_seed=RANDOM_SEED,
        all_file_names=None,
        a2br=False,
    ):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)
        self.task_list = [
            "bus.ljansbakken",
            "car.snaroya",
            "ferry.nesoddtangen",
            "metro.kalbakken",
            "norway_bus",
            "norway_car",
            "norway_metro",
            "norway_train",
            "norway_tram",
            "amazon",
            "yahoo",
            "facebook",
            "youtube",
        ]

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        if all_file_names is not None:
            self.task2idx = initialize_tasks(self.task_list, all_file_names)
        self.task_id = int(0)
        self.a2br_flag = a2br
        if self.a2br_flag:
            idx_member = self.task2idx[self.task_list[self.task_id]]
            idx_str = np.random.randint(len(idx_member))
            self.trace_idx = idx_member[idx_str]
        else:
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        # 超分相关
        self.edge_k = 0
        self.client_k = 0
        self.base_sr_delay = 1

    def get_random_gauss_val(self, K_RANGE):
        bottom, top = K_RANGE
        mean = (bottom + top) / 2
        val = random.gauss(mean, sigma)
        while val < bottom or val > top:
            val = random.gauss(mean, sigma)
        return val
    def get_k(self):
        ke = self.get_random_gauss_val(K_RANGE)
        kc = self.get_random_gauss_val(K_RANGE) / 2

        factor = random.randint(1, MAX_K_FACTOR)
        if  random.randint(0,1) == 0:
            ke *= factor
        else:
            kc *= factor / 2 # 客户端sr会减小传输的数据量，为了让place_action的动作更均匀，理应让客户端计算能力更弱些

        self.edge_k = ke
        self.client_k = kc
        return ke, kc

    def reset(self, net=True):
        if net:
            if self.a2br_flag:
                idx_member = self.task2idx[self.task_list[self.task_id]]
                idx_str = np.random.randint(len(idx_member))
                self.trace_idx = idx_member[idx_str]
            else:
                self.trace_idx = np.random.randint(len(self.all_cooked_time))
        # TODO
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.mahimahi_ptr = 1

        # self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_chunk_counter = 0
        self.buffer_size = 0
        self.get_k()
        return self.edge_k / float(K_RANGE[1]), self.client_k / float(K_RANGE[1])

    def set_trace_idx(self, idx):
        self.trace_idx = idx
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

    def set_task(self, idx):
        self.task_id = int(idx)
        idx_member = self.task2idx[self.task_list[self.task_id]]
        idx_str = np.random.randint(len(idx_member))
        self.trace_idx = idx_member[idx_str]
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        return self.task_id == len(self.task_list) - 1

    def reset_task(self):
        self.task_id = int(0)

    def get_video_size(self):
        return self.video_size

    def stash_env_state(self):
        return [self.mahimahi_ptr, self.last_mahimahi_time]

    def unstash_env_state(self, mahimahi_state):
        self.mahimahi_ptr, self.last_mahimahi_time = mahimahi_state

    def transmit_chunk(self, quality, sr_place):
        network_amplify = 2
        video_chunk_counter_sent = 0
        trans_bitrate = quality
        delay = 0
        sr_delay = 0
        if sr_place==0:
            trans_bitrate = min(BITRATE_LEVELS-1, quality+2) # 超分提升2等级的分辨率
            if trans_bitrate>quality: #实际进行了超分
                sr_delay += ((quality + BITRATE_LEVELS) / BITRATE_LEVELS) / self.edge_k * self.base_sr_delay
        video_chunk_size = self.video_size[trans_bitrate][self.video_chunk_counter]

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE * network_amplify
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (
                    (video_chunk_size - video_chunk_counter_sent)
                    / throughput
                    / PACKET_PAYLOAD_PORTION
                )
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr]
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        if sr_place>=0:
            new_bitrate = min(BITRATE_LEVELS - 1, quality + 2)
        else:
            new_bitrate = trans_bitrate
        if sr_place==1 and new_bitrate > quality:  # 实际进行了超分
                sr_delay += ((quality + BITRATE_LEVELS) / BITRATE_LEVELS) / self.client_k * self.base_sr_delay

        return delay + sr_delay, sr_delay, trans_bitrate, new_bitrate
    def get_video_chunk(self, quality, sr_place, no_sr=False, no_abr=False):
        if no_abr:
            quality = 0
        # assert quality >= 0
        assert quality < BITRATE_LEVELS

        # use the delivery opportunity in mahimahi
        if no_sr:
            if quality<0:
                quality = 0
            else:
                quality = min(quality+2, BITRATE_LEVELS-1)
            delay, sr_delay, trans_bitrate, new_bitrate = self.transmit_chunk(quality, -1)  # in ms
            place_reward = 0
        elif sr_place==-1:
            delay, sr_delay, trans_bitrate, new_bitrate = self.transmit_chunk(quality, -1)  # in ms
            place_reward = 0
        elif quality<0:
            delay, sr_delay, trans_bitrate, new_bitrate = self.transmit_chunk(0, -1)  # in ms
            place_reward = 0
        else:
            mahimahi_state_initial = self.stash_env_state()
            delay0, sr_delay0, trans_bitrate0, new_bitrate0 = self.transmit_chunk(quality, 0)  # in ms
            mahimahi_state_0 = self.stash_env_state()

            self.unstash_env_state(mahimahi_state_initial)
            delay1, sr_delay1, trans_bitrate1, new_bitrate1 = self.transmit_chunk(quality, 1)
            mahimahi_state_1 = self.stash_env_state()

            if sr_place==0:
                delay, sr_delay, trans_bitrate, new_bitrate = delay0, sr_delay0, trans_bitrate0, new_bitrate0
                place_reward = 10 * (delay1 - delay0)
                # place_reward = 1 * (self.edge_k - self.client_k)
                self.unstash_env_state(mahimahi_state_0)
            elif sr_place==1:
                delay, sr_delay, trans_bitrate, new_bitrate = delay1, sr_delay1, trans_bitrate1, new_bitrate1
                place_reward = 10 * (delay0 - delay1)
                # place_reward = 1 * (self.client_k - self.edge_k)
                self.unstash_env_state(mahimahi_state_1)
            else:
                raise ValueError

        if place_reward<0:
            place_reward /= 5
        delay *= MILLISECONDS_IN_SECOND
        sr_delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # add a multiplicative noise to the delay
        # delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - sr_delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - (delay - sr_delay), 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            sleep_time = self.buffer_size - BUFFER_THRESH
            self.buffer_size = BUFFER_THRESH
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            # drain_buffer_time = self.buffer_size - BUFFER_THRESH
            # sleep_time = (
            #     np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME)
            #     * DRAIN_BUFFER_SLEEP_TIME
            # )
            # self.buffer_size -= sleep_time
            #
            # while True:
            #     duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
            #     if duration > sleep_time / MILLISECONDS_IN_SECOND:
            #         self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
            #         break
            #     sleep_time -= duration * MILLISECONDS_IN_SECOND
            #     self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            #     self.mahimahi_ptr += 1
            #
            #     if self.mahimahi_ptr >= len(self.cooked_bw):
            #         # loop back in the beginning
            #         # note: trace file starts with time 0
            #         self.mahimahi_ptr = 1
            #         self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        TOTAL_VIDEO_CHUNCK = len(self.video_size[0]) - 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            # pick a random trace file
            if self.a2br_flag:
                idx_member = self.task2idx[self.task_list[self.task_id]]
                idx_str = np.random.randint(len(idx_member))
                self.trace_idx = idx_member[idx_str]
            else:
                self.trace_idx = np.random.randint(len(self.all_cooked_time))
                if self.trace_idx >= len(self.all_cooked_time):
                    self.trace_idx = 0
            # self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.get_k()

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return (
            delay,
            sr_delay,
            sleep_time / MILLISECONDS_IN_SECOND,
            return_buffer_size / MILLISECONDS_IN_SECOND,
            rebuf / MILLISECONDS_IN_SECOND,
            self.video_size[trans_bitrate][self.video_chunk_counter],
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
            place_reward / MAX_K_FACTOR,
            float(self.edge_k) / (2*K_RANGE[1]),
            float(self.client_k) / (2*K_RANGE[1])
        )
