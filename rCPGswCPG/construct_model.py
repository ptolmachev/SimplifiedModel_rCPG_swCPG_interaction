from rCPGswCPG.Network import Network, Neuron
import numpy as np

# reconstruct the full model of the network given the set of parameters:
# 'dt',
# 'pnames' - names of the neural populations,
# 'drives_misc' - the array of values containing drives to populations from one or more sources
# 'tau' - spike-frequency adaptation constants for the populations
# 'W' - connectivity matrix

def construct_model(model_params):
    T_transient = 15
    dt = model_params["dt"]
    pnames = model_params["pnames"]
    drives_misc = model_params["drives_misc"]
    drives = np.sum(drives_misc, axis=0)
    tau = model_params["tau"]
    W = model_params["W"]
    neuron_defaults = model_params.get("neuron_defaults", {})

    populations = [
         Neuron(
            name=pnames[i],
            drive=drives[pnames.index(pnames[i])],
            tau=tau[pnames.index(pnames[i])],
            alpha=neuron_defaults.get("alpha", 0.01),
            bias=neuron_defaults.get("bias", -0.2),
            tau_v=neuron_defaults.get("tau_v", 0.005)
         )
        for i in range(len(pnames))
    ]
    model = Network(populations=populations, dt=dt, W=W)
    model.run(T_transient, np.zeros(model.N))
    model.clear_history()
    return model


