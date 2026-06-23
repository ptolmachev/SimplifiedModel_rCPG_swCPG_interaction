import os
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf


def _as_plain_dict(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a config node into a plain Python dict.

    Args:
        cfg: Mapping-like config node.

    Returns:
        A plain dictionary with primitive Python containers.
    """
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def model_params_from_cfg(model_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the model parameter dictionary expected by construct_model.

    Args:
        model_cfg: Hydra/OmegaConf model parameter config with keys
            `dt`, `pnames`, `tau`, `connections`, `drives_misc`, and
            `neuron_defaults`.
            Connections may be specified either as the compact mapping
            `"(to, from)": weight` or the older list-of-dicts form.

    Returns:
        Dict with numpy arrays and metadata fields:
            `pnames`, `N`, `dt`, `tau`, `W`, `drives_misc`.
    """
    cfg = _as_plain_dict(model_cfg)
    pnames: List[str] = list(cfg["pnames"])
    N = len(pnames)
    pidx = {name: i for i, name in enumerate(pnames)}

    dt = float(cfg["dt"])

    tau_cfg = _as_plain_dict(cfg["tau"])
    tau = np.zeros(N)
    for name, value in tau_cfg.items():
        if name not in pidx:
            raise KeyError(f"Unknown population in tau config: {name}")
        tau[pidx[name]] = float(value)

    W = np.zeros((N, N))
    connections_cfg = cfg["connections"]
    if isinstance(connections_cfg, dict):
        connection_items = connections_cfg.items()
    else:
        connection_items = []
        for edge in connections_cfg:
            edge_dict = _as_plain_dict(edge)
            connection_items.append(((edge_dict["to"], edge_dict["from"]), edge_dict["weight"]))

    for key, weight in connection_items:
        if isinstance(key, (list, tuple)) and len(key) == 2:
            to_pop, from_pop = key
        else:
            key_str = str(key).strip()
            match = re.match(r'^\(?\s*([^,]+?)\s*,\s*([^)]+?)\s*\)?$', key_str)
            if match is None:
                raise ValueError(f"Unrecognized connection key format: {key_str}")
            to_pop, from_pop = match.group(1), match.group(2)
        to_pop = str(to_pop).strip()
        from_pop = str(from_pop).strip()
        weight = float(weight)
        if from_pop not in pidx or to_pop not in pidx:
            raise KeyError(f"Unknown population in connection: {to_pop} <- {from_pop}")
        # Canonical convention: W[to, from].
        W[pidx[to_pop], pidx[from_pop]] = weight

    neuron_defaults = _as_plain_dict(cfg["neuron_defaults"])
    defaults = {
        "alpha": float(neuron_defaults["alpha"]),
        "bias": float(neuron_defaults["bias"]),
        "tau_v": float(neuron_defaults["tau_v"]),
        "beta": float(neuron_defaults["beta"]),
    }

    drives_cfg = cfg["drives_misc"]
    if isinstance(drives_cfg, dict):
        drive_items = drives_cfg.items()
    else:
        drive_items = []
        for row in drives_cfg:
            row_dict = _as_plain_dict(row)
            for name, value in row_dict.items():
                drive_items.append(((name, f"row{len(drive_items)}"), value))

    source_to_row: Dict[str, int] = {}
    drive_rows: List[Dict[str, float]] = []
    for key, weight in drive_items:
        if isinstance(key, (list, tuple)) and len(key) == 2:
            to_pop, source_name = key
        else:
            key_str = str(key).strip()
            match = re.match(r'^\(?\s*([^,]+?)\s*,\s*([^)]+?)\s*\)?$', key_str)
            if match is None:
                raise ValueError(f"Unrecognized drive key format: {key_str}")
            to_pop, source_name = match.group(1), match.group(2)
        to_pop = str(to_pop).strip()
        source_name = str(source_name).strip()
        weight = float(weight)
        if to_pop not in pidx:
            raise KeyError(f"Unknown population in drives_misc target: {to_pop}")
        row_idx = source_to_row.get(source_name)
        if row_idx is None:
            row_idx = len(drive_rows)
            source_to_row[source_name] = row_idx
            drive_rows.append({})
        drive_rows[row_idx][to_pop] = weight

    drives_misc = np.zeros((len(drive_rows), N))
    for row_idx, row in enumerate(drive_rows):
        for name, value in row.items():
            drives_misc[row_idx, pidx[name]] = float(value)

    return {
        "pnames": pnames,
        "N": N,
        "dt": dt,
        "tau": tau,
        "W": W,
        "drives_misc": drives_misc,
        "neuron_defaults": defaults,
    }


def load_model_cfg_file(model_name: str, config_root: Optional[str] = None) -> DictConfig:
    """Load a model parameter config file by name.

    Args:
        model_name: Config basename in `configs/model_params` without `.yaml`.
        config_root: Optional absolute path to config root; defaults to
            `<repo>/configs`.

    Returns:
        OmegaConf DictConfig for the model.
    """
    if config_root is None:
        config_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "configs",
        )
    model_cfg_path = os.path.join(config_root, "model_params", f"{model_name}.yaml")
    if not os.path.exists(model_cfg_path):
        raise FileNotFoundError(f"Model config not found: {model_cfg_path}")
    return OmegaConf.load(model_cfg_path)
