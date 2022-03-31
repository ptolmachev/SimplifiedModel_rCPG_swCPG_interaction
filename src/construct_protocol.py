from src.exp_protocols.protocols import *

def construct_protocol(model, name, params):
    if name == "Protocol_noSI":
        protocol = Protocol_noSI(model, **params)
    elif name == "Protocol_longSI":
        protocol = Protocol_longSI(model, **params)
    elif name == "Protocol_shortSI":
        protocol = Protocol_shortSI(model, **params)
    else:
        raise NameError("No such protocol is currently implemented.")
    return protocol
