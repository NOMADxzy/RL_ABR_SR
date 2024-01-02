# RL_ABR_SR

Re-implementation of existing neural ABR algorithms with Pytorch, MAML.

## User guide

### To Train

```python main.py```

### To Test

```python client.py```

>Note that the original version of Pensieve using asynchronous advantage actor-critic algorithm (A3C) to train the policy, which can only implementated on CPU. Our A2C version removes the asynchronous setting and use GPU to accelerate the speed of NNs training. 

Further improvements are ongoing...
