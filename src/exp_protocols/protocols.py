from src.utils.gen_utils import modify_array
import numpy as np

class Protocol():
    def __init__(self, model):
        self.model = model
        self.external_inputs = np.zeros(len(model.pnames))

    def run(self):
        self.model.clear_history()
        return None

class Protocol_noSI(Protocol):
    def __init__(self, model, T):
        super().__init__(model)
        self.name = "Protocol_noSI"
        self.T = T

    def run(self):
        super().run()
        i = self.model.pnames.index("Sensory_relay")
        self.model.run(self.T, input=modify_array(self.external_inputs, i, 0))
        return None

class Protocol_longSI(Protocol):
    def __init__(self, model, T, amp):
        super().__init__(model)
        self.name = "Protocol_longSI"
        self.T = T
        self.amp = amp

    def run(self):
        super().run()
        i = self.model.pnames.index("Sensory_relay")
        self.model.run(self.T, input=modify_array(self.external_inputs, i, self.amp))
        return None

class Protocol_shortSI(Protocol):
    def __init__(self, model, interim_T, amp, stim_duration):
        super().__init__(model)
        self.name = "Protocol_shortSI"
        self.interim_T = interim_T
        self.stim_duration = stim_duration
        self.amp = amp

    def run(self):
        super().run()
        i = self.model.pnames.index("Sensory_relay")
        self.model.run(self.interim_T, input=modify_array(self.external_inputs, i, 0.0))
        self.model.run(self.stim_duration, input=modify_array(self.external_inputs, i , self.amp))
        self.model.run(self.interim_T + 2 * np.random.randn(), input=modify_array(self.external_inputs, i, 0.0))
        self.model.run(self.stim_duration, input=modify_array(self.external_inputs, i, self.amp))
        self.model.run(self.interim_T + 2 * np.random.randn(), input=modify_array(self.external_inputs, i, 0.0))
        self.model.run(self.stim_duration, input=modify_array(self.external_inputs, i, self.amp))
        self.model.run(self.interim_T + 2 * np.random.randn(), input=modify_array(self.external_inputs, i, 0.0))
        self.model.run(self.stim_duration, input=modify_array(self.external_inputs, i, self.amp))
        self.model.run(5, input=modify_array(self.external_inputs, i, 0.0))
        return None


def run_full_protocol(model):
    amp = 0.45
    stim_duration = 0.1
    T = 5
    T_long_stim = 10
    T_transient = 15
    pnames = model.pnames
    external_inputs = np.zeros(len(pnames))
    model.run(T_transient, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0))
    model.clear_history()
    model.run(3 * T, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0))
    model.run(T_long_stim, input=modify_array(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(T, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0.0))
    model.run(stim_duration, input=modify_array(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(T + 2 * np.random.randn(), input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0.0))
    model.run(stim_duration, input=modify_array(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(T + 2 * np.random.randn(), input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0.0))
    model.run(stim_duration, input=modify_array(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(T + 2 * np.random.randn(), input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0.0))
    model.run(stim_duration, input=modify_array(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(5, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0.0))
    return None

def run_KF_inhibited_protocol(model):
    amp = 0.45
    stim_duration = 0.1
    T = 5
    T_transient = 15
    T_long_stim = 10
    pnames = model.pnames

    #set parameters for the inhibition of the KF:
    KF_populations = ["KF_gate", "KF_phasic"]
    for KF_pop in KF_populations:
        model.populations[pnames.index(KF_pop)].drive = 0
        for name in pnames:
            model.W[pnames.index(name), pnames.index(KF_pop)] = 0.0
            model.W[pnames.index(KF_pop), pnames.index(name)] = 0.0
    model.populations[pnames.index("Exp")].drive = 0.0
    model.populations[pnames.index("Insp")].drive = 0.3

    external_inputs = np.zeros(len(pnames))
    model.run(T_transient, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0))
    model.clear_history()

    model.run(3 * T, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0))
    model.run(T_long_stim, input=modify_array(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(T, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0.0))
    model.run(stim_duration, input=modify_array(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(T + 2 * np.random.randn(), input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0.0))
    model.run(stim_duration, input=modify_array(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(T + 2 * np.random.randn(), input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0.0))
    model.run(stim_duration, input=modify_array(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(T + 2 * np.random.randn(), input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0.0))
    model.run(stim_duration, input=modify_array(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(5, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0.0))
    return None
