import numpy as np
from collections import deque
from copy import deepcopy

def firing_rate(v):
    return 1.0 / (1 + np.exp(-0.3 * v))

class Neuron():
    '''
    A neuron class with simple continuous dynamics exhibiting firing rate adaptation
    '''
    def __init__(self, name, dt, drive, tau):
        self.name = name
        self.dt = dt
        self.state = np.zeros(2)
        self.state_history = deque()
        self.drive = drive
        self.tau = tau

    def rhs(self, inp):
        v, m = self.state
        rhs_v = 200 * (-0.01 * v - m + (self.drive - 0.2) + inp)
        rhs_m = (self.fr() - m) / self.tau
        return np.array([rhs_v, rhs_m])

    def fr(self):
        '''
        Calculates firing rate of a neuron based on its average 'voltage'
        '''
        return firing_rate(self.state[0])

    def get_next_state(self, inp):
        state = self.state + self.dt * self.rhs(inp)
        return state

    def get_state_history(self):
        return np.array(self.state_history)

    def update_history(self):
        self.state_history.append(deepcopy(self.state))
        return None

    def step(self, inp):
        self.state += self.dt * self.rhs(inp)
        self.update_history()
        return None

    def run(self, T, inp):
        for i in range(int(T * 1000 / self.dt)):
            self.step(inp)
        return None

# class BurstingNeuron(Neuron):
#     def __init__(self, name, dt, drive, tau):
#         super().__init__(name, dt, drive, tau)
#
#     def rhs(self, inp):
#         v, h = self.state
#         alpha = 0.1
#         beta = 0.001
#         lmbd = 1
#         b = 7.0
#         rhs_v = (alpha * v - beta * v**3 - h + 10 * self.drive)
#         rhs_h = (lmbd * v + b - h) / self.tau
#         return np.array([rhs_v, rhs_h])

class BurstingNeuron(Neuron):
    def __init__(self, name, dt, drive, tau):
        super().__init__(name, dt, drive, tau)

    def rhs(self, inp):
        v, h = self.state
        p_inf = 1.0 / (1 + np.exp(-0.2 * (v + 15)))
        h_inf = 1.0 / (1 + np.exp(0.2 * (v + 30)))
        rhs_v = 100 * (-0.01 * v + 2 * p_inf * h + (self.drive - 0.3) + inp)
        rhs_h = (h_inf - h) / self.tau
        return np.array([rhs_v, rhs_h])


class Network():
    def __init__(self, populations, dt, W):
        self.W = W
        self.populations = populations
        self.N = len(self.populations)
        self.dt = dt
        # dt should be the same for all of the populations
        for i in range(self.N):
            self.populations[i].dt = self.dt
        self.pnames = [self.populations[i].name for i in range(self.N)]

    def get_fr(self):
        '''
        Combines firing rates of all neural populations into one array for the subsequent processing
        '''
        fr = np.array([self.populations[i].fr() for i in range(self.N)])
        return fr

    def calc_synaptic_inputs(self):
        fr = self.get_fr()
        synaptic_inputs = (self.W.T @ fr.reshape(-1, 1)).flatten()
        return synaptic_inputs

    def step(self, external_inputs):
        synaptic_inputs = self.calc_synaptic_inputs()
        for i in range(self.N):
            self.populations[i].step(synaptic_inputs[i] + external_inputs[i])
        return None

    def run(self, T, input):
        T_steps =  int(np.ceil(np.float(T)*1000/self.dt))
        for i in range(T_steps):
            self.step(input)

    def get_raw_history(self):
        v_history = np.array(np.hstack([self.populations[i].get_state_history()[:, 0].reshape(-1, 1) for i in range(self.N)]))
        return v_history

    def get_recordings(self):
        v_history = self.get_raw_history()
        fr_history = firing_rate(v_history)
        t = (self.dt * np.arange(fr_history.shape[0]) / 1000)  # in sec
        recordings = dict()
        recordings['population_names'] = self.pnames
        recordings["fr_history"] = fr_history
        recordings["t"] = t
        recordings['dt'] = self.dt
        return recordings

    def clear_history(self):
        for i in range(self.N):
            self.populations[i].state_history = deque()
        return None


if __name__ == '__main__':
    # simple demo
    from matplotlib import pyplot as plt
    dt = 0.1
    drives = np.ones(2)
    nrn1 = Neuron(name="1", dt=dt, drive=drives[0], tau=1000)
    nrn2 = Neuron(name="2", dt=dt, drive=drives[1], tau=1000)
    W = np.array([[0, -0.7] ,[-0.5, 0]])
    net = Network(populations=[nrn1, nrn2], dt=dt, W=W)
    net.run(T=10, input=np.zeros(2))
    data_dict = net.get_recordings()
    plt.plot(data_dict['t'], data_dict["fr_history"])
    plt.show()







