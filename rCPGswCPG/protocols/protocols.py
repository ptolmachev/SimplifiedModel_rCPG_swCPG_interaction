from rCPGswCPG.utils.gen_utils import put
import numpy as np

class Protocol():
    def __init__(self, model):
        self.model = model
        self.external_inputs = np.zeros(len(model.populations))

    def run(self):
        self.model.clear_history()
        return None

class Protocol_noSI(Protocol):
    ''' Protocol without Sensory Input (SI)'''
    def __init__(self, model, T):
        super().__init__(model)
        self.name = "Protocol_noSI"
        self.T = T

    def run(self):
        super().run()
        i = self.model.pnames.index("Sensory_relay")
        self.model.run(self.T, input=put(self.external_inputs, i, 0))
        return None

class Protocol_longSI(Protocol):
    ''' Protocol with long Sensory Input '''
    def __init__(self, model, T, amp):
        super().__init__(model)
        self.name = "Protocol_longSI"
        self.T = T
        self.amp = amp

    def run(self):
        super().run()
        i = self.model.pnames.index("Sensory_relay")
        self.model.run(self.T, input=put(self.external_inputs, i, self.amp))
        return None

class Protocol_shortSI(Protocol):
    ''' Protocol with n_stim short Sensory Input '''
    def __init__(self, model, interim_T, amp, stim_duration, n_stim=4):
        super().__init__(model)
        self.name = "Protocol_shortSI"
        self.interim_T = interim_T
        self.stim_duration = stim_duration
        self.amp = amp
        self.n_stim = n_stim

    def run(self):
        super().run()
        i = self.model.pnames.index("Sensory_relay")
        for n in range(self.n_stim):
            self.model.run(self.interim_T + int(n > 0) * 2 * np.random.randn(), input=put(self.external_inputs, i, 0.0))
            self.model.run(self.stim_duration, input=put(self.external_inputs, i, self.amp))
        self.model.run(5, input=put(self.external_inputs, i, 0.0))
        return None

class Protocol_LongShortSI(Protocol):
    ''' Protocol with one long Sensory Input followed by n_stim short Sensory Input '''
    def __init__(self, model, noSI_T=2, longSI_T=10, interim_T=5, amp=0.45, stim_duration=0.1, n_stim=1):
        super().__init__(model)
        self.name = "Protocol_LongShortSI"
        self.noSI_T = noSI_T
        self.longSI_T = longSI_T
        self.interim_T = interim_T
        self.stim_duration = stim_duration
        self.amp = amp
        self.n_stim = n_stim

    def run(self):
        super().run()
        i = self.model.pnames.index("Sensory_relay")
        self.model.run(self.noSI_T, input=put(self.external_inputs, i, 0.0))
        self.model.run(self.longSI_T, input=put(self.external_inputs, i, self.amp))
        for n in range(self.n_stim):
            self.model.run(self.interim_T + int(n > 0) * 2 * np.random.randn(), input=put(self.external_inputs, i, 0.0))
            self.model.run(self.stim_duration, input=put(self.external_inputs, i, self.amp))
        self.model.run(self.noSI_T, input=put(self.external_inputs, i, 0.0))
        return None

def run_KF_inhibited_protocol(model):
    ''' Protocol with inhibited KF (Pons) population(s) '''
    amp = 0.45
    stim_duration = 0.1
    T = 5
    T_transient = 15
    T_long_stim = 10
    pnames = model.populations

    #set parameters for the inhibition of the KF (Pons):
    KF_populations = ["KF_gate", "KF_phasic"]
    for KF_pop in KF_populations:
        model.populations[pnames.index(KF_pop)].drive = 0
        for name in pnames:
            model.W[pnames.index(name), pnames.index(KF_pop)] = 0.0
            model.W[pnames.index(KF_pop), pnames.index(name)] = 0.0
    model.populations[pnames.index("Exp")].drive = 0.0
    model.populations[pnames.index("Insp")].drive = 0.3

    external_inputs = np.zeros(len(pnames))
    model.run(T_transient, input=put(external_inputs, pnames.index("Sensory_relay"), 0))
    model.clear_history()

    model.run(3 * T, input=put(external_inputs, pnames.index("Sensory_relay"), 0))
    model.run(T_long_stim, input=put(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(T, input=put(external_inputs, pnames.index("Sensory_relay"), 0.0))
    model.run(stim_duration, input=put(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(T + 2 * np.random.randn(), input=put(external_inputs, pnames.index("Sensory_relay"), 0.0))
    model.run(stim_duration, input=put(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(T + 2 * np.random.randn(), input=put(external_inputs, pnames.index("Sensory_relay"), 0.0))
    model.run(stim_duration, input=put(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(T + 2 * np.random.randn(), input=put(external_inputs, pnames.index("Sensory_relay"), 0.0))
    model.run(stim_duration, input=put(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(5, input=put(external_inputs, pnames.index("Sensory_relay"), 0.0))
    return None


def run_standalone_protocol(model, amp = 0.40, stim_duration=0.1, T_transient=15, T_no_stim=30, T_long_stim=10, interim_T=10, n_stim=4):
    pnames = model.pnames
    external_inputs = np.zeros(len(pnames))
    model.run(T_transient, input=put(external_inputs, pnames.index("Sensory_relay"), 0))
    model.clear_history()

    i = model.pnames.index("Sensory_relay")
    # no SI
    model.run(T_no_stim, input=put(external_inputs, i, 0))
    # long SI
    model.run(T_long_stim, input=put(external_inputs, i, amp))
    # short SI
    for n in range(n_stim):
        model.run(interim_T + int(n>0) * 2 * np.random.randn(), input=put(external_inputs, i, 0.0))
        model.run(stim_duration, input=put(external_inputs, i, amp))
    model.run(5, input=put(external_inputs, i, 0.0))
    return None
