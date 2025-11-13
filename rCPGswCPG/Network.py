import numpy as np
from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt

def firing_rate(v):
    return 1.0/(1.0 + np.exp(-0.3 * v)) 

class Neuron():
    '''
    A neuron class with simple continuous dynamics exhibiting firing rate adaptation
    '''
    def __init__(self, name, drive, tau):
        self.name = name
        self.state = np.zeros(2)
        self.state_history = deque()
        self.drive = drive
        self.tau = tau
        self.alpha = 0.01
        self.bias = -0.2
        self.tau_v = 0.005

    def rhs(self, inp):
        v, m = self.state
        rhs_v = (-self.alpha * v - m + (self.drive + self.bias) + inp) / self.tau_v
        rhs_m = (self.fr() - m) / self.tau
        return np.array([rhs_v, rhs_m])

    def fr(self):
        return firing_rate(self.state[0])

    def get_state_history(self):
        return np.array(self.state_history)

    def update_history(self):
        self.state_history.append(deepcopy(self.state))
        return None

    def step(self, dt, inp):
        self.state += dt * self.rhs(inp)
        self.update_history()
        return None

# class BurstingNeuron(Neuron):
#     def __init__(self, name, dt, drive, tau):
#         super().__init__(name, dt, drive, tau)

#     def rhs(self, inp):
#         v, h = self.state
#         p_inf = 1.0 / (1 + np.exp(-0.2 * (v + 15)))
#         h_inf = 1.0 / (1 + np.exp(0.2 * (v + 30)))
#         rhs_v = 100 * (-0.01 * v + 2 * p_inf * h + (self.drive - 0.3) + inp)
#         rhs_h = (h_inf - h) / self.tau
#         return np.array([rhs_v, rhs_h])

# class Network():
#     def __init__(self, populations, dt, W):
#         self.W = W
#         self.populations = populations
#         self.N = len(self.populations)
#         self.dt = dt
#         self.pnames = [self.populations[i].name for i in range(self.N)]

#     def get_fr(self):
#         fr = np.array([self.populations[i].fr() for i in range(self.N)])
#         return fr

#     def get_synaptic_inputs(self):
#         fr = self.get_fr()
#         synaptic_inputs = (self.W.T @ fr.reshape(-1, 1)).flatten()
#         return synaptic_inputs

#     def step(self, external_inputs):
#         inputs = self.get_synaptic_inputs() + external_inputs
#         for i in range(self.N):
#             self.populations[i].step(dt=self.dt, inp=inputs[i])
#         return None

#     def run(self, T, input):
#         T_steps = int(np.ceil(float(T) * 1000 / self.dt))
#         for i in range(T_steps):
#             self.step(input)
#         return None

#     def get_raw_history(self):
#         v_history = np.array(np.hstack([self.populations[i].get_state_history()[:, 0].reshape(-1, 1) for i in range(self.N)]))
#         m_history = np.array(np.hstack([self.populations[i].get_state_history()[:, 1].reshape(-1, 1) for i in range(self.N)]))
#         return v_history, m_history

#     def get_recordings(self):
#         v_history, m_history = self.get_raw_history()
#         fr_history = firing_rate(v_history)
#         t = (self.dt * np.arange(fr_history.shape[0]) / 1000)  # in sec
#         recordings = dict()
#         recordings['population_names'] = self.pnames
#         recordings["fr_history"] = fr_history
#         recordings["v_history"] = v_history
#         recordings["m_history"] = m_history
#         recordings["t"] = t
#         recordings['dt'] = self.dt
#         return recordings

#     def clear_history(self):
#         for i in range(self.N):
#             self.populations[i].state_history = deque()
        # return None

def firing_rate(x):
    return 1.0 / (1.0 + np.exp(-0.3 * x))

class Network:
    def __init__(self, populations, dt, W):
        self.populations = populations
        self.N = len(populations)
        self.dt = float(dt)
        self.W = np.asarray(W, dtype=float)
        self.WT = self.W.T

        # pull per-neuron constants once
        self.pnames = [p.name for p in populations]
        self.drive  = np.array([p.drive for p in populations], dtype=float)
        self.tau    = np.array([p.tau   for p in populations], dtype=float)
        self.alpha  = populations[0].alpha if hasattr(populations[0], 'alpha') else 0.01
        self.bias   = populations[0].bias  if hasattr(populations[0], 'bias')  else -0.2
        self.tau_v  = populations[0].tau_v if hasattr(populations[0], 'tau_v') else 0.005

        # state (v, m) kept vectorized; also mirror into each population for compatibility
        self.v = np.zeros(self.N, dtype=float)
        self.m = np.zeros(self.N, dtype=float)
        for i, p in enumerate(populations):
            p.state = np.array([0.0, 0.0], dtype=float)
            p.state_history = deque()

        # step history for get_raw_history(); append per step
        self._v_hist = []
        self._m_hist = []

    def get_fr(self):
        return firing_rate(self.v)

    def get_synaptic_inputs(self):
        return (self.WT @ self.get_fr().reshape(-1, 1)).ravel()

    def _push_history(self):
        # keep vectorized buffers
        self._v_hist.append(self.v.copy())
        self._m_hist.append(self.m.copy())
        # and mirror into per-population deques for strict compatibility
        for i, p in enumerate(self.populations):
            p.state = np.array([self.v[i], self.m[i]])
            p.state_history.append(p.state.copy())

    def step(self, external_inputs):
        ext = external_inputs
        if np.isscalar(ext):
            ext = float(ext) * np.ones(self.N)
        else:
            ext = np.asarray(ext, dtype=float)

        dt   = self.dt
        v0   = self.v
        m0   = self.m

        def rhs(v, m):
            fr  = firing_rate(v)
            syn = self.WT @ fr
            rv  = (-self.alpha * v - m + (self.drive + self.bias) + syn + ext) / self.tau_v
            rm  = (fr - m) / self.tau
            return rv, rm

        k1v, k1m = rhs(v0,                     m0)
        k2v, k2m = rhs(v0 + 0.5 * dt * k1v,    m0 + 0.5 * dt * k1m)
        k3v, k3m = rhs(v0 + 0.5 * dt * k2v,    m0 + 0.5 * dt * k2m)
        k4v, k4m = rhs(v0 + dt * k3v,          m0 + dt * k3m)

        self.v = v0 + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        self.m = m0 + (dt / 6.0) * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)

        self._push_history()
        return None


    def run(self, T, input):
        T_steps = int(np.ceil(float(T) * 1000.0 / self.dt))
        for _ in range(T_steps):
            self.step(input)
        return None

    def get_raw_history(self):
        v_history = np.vstack(self._v_hist) if self._v_hist else np.zeros((0, self.N))
        m_history = np.vstack(self._m_hist) if self._m_hist else np.zeros((0, self.N))
        return v_history, m_history

    def get_recordings(self):
        v_history, m_history = self.get_raw_history()
        fr_history = firing_rate(v_history) if v_history.size else v_history
        t = (self.dt * np.arange(fr_history.shape[0])) / 1000.0
        return {
            'population_names': self.pnames,
            'fr_history': fr_history,
            'v_history': v_history,
            'm_history': m_history,
            't': t,
            'dt': self.dt
        }

    def clear_history(self):
        self._v_hist.clear()
        self._m_hist.clear()
        for p in self.populations:
            p.state_history = deque()
        return None
        
    def sync_params(self):
        self.W  = np.asarray(self.W, dtype=np.float64)  # force stable dtype
        self.WT = self.W.T
        self.drive = np.array([p.drive for p in self.populations], dtype=np.float64)
        self.tau   = np.array([p.tau   for p in self.populations], dtype=np.float64)
        self.tau_v = np.array([p.tau_v for p in self.populations], dtype=np.float64)


if __name__ == '__main__':
    # simple demo
    dt = 0.75
    drives = np.ones(2)
    nrn1 = Neuron(name="1", drive=drives[0], tau=1500)
    nrn2 = Neuron(name="2", drive=drives[1], tau=1500)
    W = np.array([[0, -0.7] ,[-0.5, 0]])
    net = Network(populations=[nrn1, nrn2], dt=dt, W=W)
    net.run(T=10, input=np.zeros(2))
    data_dict = net.get_recordings()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(data_dict['t'], data_dict["fr_history"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(['population1', 'population2'], loc=(0.75, 0.75), frameon=False)
    plt.show()





