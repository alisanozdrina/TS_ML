#!/usr/bin/env python3
"""Build ML HDF5 datasets from RNO-G ROOT run directories.

Inspection notes from the RNOG examples:

1. Reader: the RNOG examples use
   ``NuRadioReco.modules.RNO_G.dataProviderRNOG.dataProviderRNOG`` for ROOT
   data. That provider wraps ``readRNOGDataMattak`` and expects either mattak
   run directories or combined ROOT files.
2. Calibrated waveforms: ``dataProviderRNOG`` applies voltage conversion in
   ``readRNOGDataMattak`` by default, then block-offset removal, glitch
   detection, and cable-delay subtraction. The standard examples additionally
   call ``NuRadioReco.examples.RNOG.processing.process_event`` for resampling,
   bandpass filtering, phase-only hardware response correction, CW notch
   filtering, and ``channelSignalReconstructor``.
3. Metadata: run and event number come from ``event.get_run_number()`` and
   ``event.get_id()``. Timestamp is normally ``station.get_station_time().unix``.
   Trigger name/type is available through ``station.get_first_trigger()`` when
   a trigger fired; this script stores a configurable integer code with -1 as
   the missing sentinel. The HDF5 output also stores per-channel waveform timing
   in ``trace_start_time`` and ``trace_delta_t`` so sample times can be
   reconstructed as ``trace_start_time[event, channel] + i *
   trace_delta_t[event, channel]``.
4. Features: ``snr_avg_deep`` is computed with the local
   ``channelFeatureExtractor`` over the deep channels. ``coherent_snr_deep``
   follows the project feature example's coherent-sum convention using
   ``trace_utilities.get_coherent_sum``. ``passed_hit_filter`` comes from
   ``stationHitFilter``. The RNOG feature example does not compute
   ``max_corr``; the optional ``reco3d`` block can run
   ``InterferometricReco3D`` here and write ``rho``, ``phi``, ``z``,
   ``max_corr``, and ``surf_corr_zen``. Until that block is enabled,
   reco quantities are written as NaN.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import inspect
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import h5py
import numpy as np
import yaml


LOG = logging.getLogger("build_ml_run_h5")

N_CHANNELS = 24
N_SAMPLES = 2048
DEFAULT_WAVEFORM_SHAPE = (N_CHANNELS, N_SAMPLES)

VPOL_CHS = (0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23)
HPOL_CHS = (4, 8, 11, 21)
PA_CHS = (0, 1, 2, 3)
DEEP_CHS = tuple(sorted(set(VPOL_CHS) | set(HPOL_CHS)))

FEATURE_DTYPES = {
    "snr_avg_deep": "f4",
    "coherent_spectral_centroid_pa": "f4",
    "n_coincident_pairs_deep": "i4",
    "coherent_spectral_peak_frequency_pa": "f4",
    "coherent_spectral_entropy_pa": "f4",
    "coherent_snr_deep": "f4",
    "max_amplitude_avg_pa_over_hpol": "f4",
    "impulse_corr_bipolar_wide_avg_pa": "f4",
    "impulse_corr_bipolar_avg_pa": "f4",
    "surf_corr_zen": "f4",
    "rho": "f4",
    "phi": "f4",
    "z": "f4",
    "max_corr": "f4",
    "passed_hit_filter": "i1",
}
REQUIRED_EXTRACTOR_GROUPS = ("snr", "max_amplitude", "impulse_correlations")

TRIGGER_CODE = {
    "LT": 1,
    "FORCE": 2,
    "RADIANT": 3,
    "UNKNOWN": 4,
}
SOURCE_TYPE_CODE = {
    "data": 0,
    "simulation": 1,
    "sim": 1,
}
MISSING_INT8 = np.int8(-1)
MISSING_FLOAT = np.float32(np.nan)


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in list(cfg.items()):
        if isinstance(value, str):
            cfg[key] = os.path.expandvars(value)
    return cfg


def configure_nuradio_path(config: Mapping) -> None:
    """Optionally prepend a local NuRadioMC checkout to sys.path."""
    for key in ("nuradio_path", "nuradiomc_path"):
        root = config.get(key)
        if root:
            root = str(Path(root).expanduser().resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            return


def discover_input_files(input_dir: Path, station: int, run: int) -> List[str]:
    """Return provider inputs for one run.

    Prefer mattak run directories because ``readRNOGDataMattak`` can then find
    ``waveforms.root`` plus the needed header/status/pedestal companions.
    Combined ROOT files are accepted as a fallback.
    """
    station_tokens = (f"station{station}", f"station{station:02d}")
    run_tokens = (f"run{run}", f"run{run:06d}", f"run{run:04d}")

    candidates = []
    for st in station_tokens:
        for rn in run_tokens:
            candidates.append(input_dir / st / rn)
            candidates.append(input_dir / st / str(run) / rn)

    for path in candidates:
        if path.is_dir():
            if (path / "combined.root").exists():
                return [str(path)]
            if (path / "waveforms.root").exists() and (path / "headers.root").exists():
                missing = [
                    name for name in ("daqstatus.root", "pedestal.root")
                    if not (path / name).exists()
                ]
                if missing:
                    LOG.warning(
                        "Run directory %s is missing %s; dataProviderRNOG may skip it.",
                        path, ", ".join(missing),
                    )
                return [str(path)]

    root_candidates = []
    for st in station_tokens:
        for rn in run_tokens:
            root_candidates.extend([
                input_dir / st / f"{rn}.root",
                input_dir / st / rn / "combined.root",
            ])
    for path in root_candidates:
        if path.exists():
            return [str(path)]

    raise FileNotFoundError(
        f"Could not find ROOT inputs for station {station}, run {run} under {input_dir}"
    )


def expand_input_files(patterns: Iterable[str]) -> List[str]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern) if any(c in pattern for c in "*?[]") else [pattern]
        paths.extend(Path(match).expanduser().resolve() for match in matches)
    unique = sorted(dict.fromkeys(str(path) for path in paths if path.exists()))
    if not unique:
        raise FileNotFoundError(f"No input files matched {list(patterns)}")
    return unique


def init_detector(config: Mapping):
    source = config.get("detector_source", "rnog_mongo")
    detector_file = config.get("detector_file")
    if source == "rnog_file":
        from NuRadioReco.detector.RNO_G import rnog_detector

        return rnog_detector.Detector(detector_file=detector_file)
    if source == "json":
        from NuRadioReco.detector import detector

        return detector.Detector(json_filename=detector_file)
    from NuRadioReco.detector import detector

    return detector.Detector(source="rnog_mongo")


def init_provider(input_files: List[str], det, config: Mapping):
    reader_kwargs = dict(config.get("reader_kwargs", {}) or {})
    is_nur = all(str(path).lower().endswith(".nur") for path in input_files)
    if is_nur:
        from NuRadioReco.modules.RNO_G.dataProviderNuRadio import dataProviderNuRadio

        provider = dataProviderNuRadio()
    else:
        from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG

        mattak_kwargs = {
            "read_daq_status": False,
            "read_run_info": False,
            "backend": "uproot",
        }
        mattak_kwargs.update(reader_kwargs.get("mattak_kwargs") or {})
        reader_kwargs["mattak_kwargs"] = mattak_kwargs
        provider = dataProviderRNOG()
    begin_kwargs = {"reader_kwargs": reader_kwargs}
    if "preprocessor_config" in inspect.signature(provider.begin).parameters:
        begin_kwargs["preprocessor_config"] = config.get("preprocessor", None)
    provider.begin(input_files, det=det, **begin_kwargs)
    return provider


def maybe_process_event(event, det, config: Mapping) -> None:
    if not config.get("apply_standard_processing", True):
        return
    from NuRadioReco.examples.RNOG.processing import process_event

    process_event(event, det)


def maybe_resample_output(event, station, det, config: Mapping) -> None:
    rate_ghz = config.get("output_resample_rate_ghz")
    if rate_ghz is None:
        return
    import NuRadioReco.modules.channelResampler
    from NuRadioReco.utilities import units

    resampler = NuRadioReco.modules.channelResampler.channelResampler()
    resampler.begin()
    resampler.run(event, station, det, sampling_rate=float(rate_ghz) * units.GHz)


def extract_waveforms(station, shape: Tuple[int, int] = DEFAULT_WAVEFORM_SHAPE) -> np.ndarray:
    arr = np.empty(shape, dtype=np.float32)
    for ch_id in range(shape[0]):
        if not station.has_channel(ch_id):
            raise ValueError(f"station is missing channel {ch_id}")
        trace = np.asarray(station.get_channel(ch_id).get_trace(), dtype=np.float32)
        if trace.shape != (shape[1],):
            raise ValueError(
                f"channel {ch_id} has trace shape {trace.shape}, expected {(shape[1],)}"
            )
        arr[ch_id] = trace
    return arr


def extract_waveform_timing(
    station, n_channels: int = N_CHANNELS
) -> Tuple[np.ndarray, np.ndarray]:
    trace_start_time = np.empty((n_channels,), dtype=np.float64)
    trace_delta_t = np.empty((n_channels,), dtype=np.float64)
    for ch_id in range(n_channels):
        if not station.has_channel(ch_id):
            raise ValueError(f"station is missing channel {ch_id}")
        channel = station.get_channel(ch_id)
        sampling_rate = channel.get_sampling_rate()
        if sampling_rate is None or sampling_rate == 0:
            raise ValueError(f"channel {ch_id} has invalid sampling rate {sampling_rate}")
        trace_start_time[ch_id] = float(channel.get_trace_start_time())
        trace_delta_t[ch_id] = 1.0 / float(sampling_rate)
    return trace_start_time, trace_delta_t


def get_timestamp(station) -> Optional[float]:
    try:
        station_time = station.get_station_time()
        return None if station_time is None else float(station_time.unix)
    except Exception:
        return None


def get_trigger_code(station) -> np.int8:
    try:
        trigger = station.get_first_trigger()
        if trigger is None:
            return MISSING_INT8
        name = trigger.get_name()
        if name is None:
            name = trigger.get_type()
        name = str(name)
        if name.startswith("RADIANT"):
            name = "RADIANT"
        return np.int8(TRIGGER_CODE.get(name, MISSING_INT8))
    except Exception:
        return MISSING_INT8


def get_source_type_code(config: Mapping) -> np.int8:
    source_type = config.get("source_type", "data")
    if isinstance(source_type, str):
        key = source_type.strip().lower()
        if key not in SOURCE_TYPE_CODE:
            raise ValueError(
                f"unknown source_type {source_type!r}; expected one of "
                f"{sorted(SOURCE_TYPE_CODE)} or an integer code"
            )
        return np.int8(SOURCE_TYPE_CODE[key])
    return np.int8(source_type)


def paired_sim_truth_path(nur_path: str) -> Optional[Path]:
    path = Path(nur_path)
    candidate = path.with_suffix(".hdf5")
    return candidate if candidate.exists() else None


def load_sim_truth(path: Optional[Path], station_id: int) -> Dict[str, np.ndarray]:
    if path is None:
        return {}

    wanted = {
        "energies": "energy",
        "shower_energies": "shower_energy",
        "zeniths": "zenith",
        "azimuths": "azimuth",
        "weights": "weight",
        "flavors": "flavor",
        "inelasticity": "inelasticity",
        "xx": "x",
        "yy": "y",
        "zz": "z",
        "triggered": "triggered",
        "vertex_times": "vertex_time",
    }
    truth: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:
        for source, target in wanted.items():
            if source in h5:
                truth[target] = np.asarray(h5[source][:])
        station_group = f"station_{station_id}"
        if station_group in h5:
            grp = h5[station_group]
            station_wanted = {
                "triggered": "station_triggered",
                "triggered_per_event": "station_triggered_per_event",
                "trigger_times": "station_trigger_time",
                "trigger_times_per_event": "station_trigger_time_per_event",
                "event_group_ids": "station_event_group_id",
                "event_ids": "station_event_id",
                "maximum_amplitudes": "station_maximum_amplitudes",
                "maximum_amplitudes_envelope": "station_maximum_amplitudes_envelope",
            }
            for source, target in station_wanted.items():
                if source in grp:
                    truth[target] = np.asarray(grp[source][:])
    return truth


def sim_truth_for_event(truth: Mapping[str, np.ndarray], event_index: int) -> Dict[str, np.number]:
    row = {}
    for key, values in truth.items():
        if event_index >= len(values):
            continue
        value = values[event_index]
        if np.asarray(value).shape == ():
            row[key] = np.asarray(value).item()
        elif np.asarray(value).size == 1:
            row[key] = np.asarray(value).reshape(-1)[0].item()
        else:
            row[key] = np.asarray(value)
    return row


def init_feature_tools(config: Mapping):
    from NuRadioReco.modules.channelFeatureExtractor import channelFeatureExtractor
    from NuRadioReco.modules.RNO_G.stationHitFilter import stationHitFilter

    extractor = channelFeatureExtractor()
    feature_cfg = dict(config.get("features", {}) or {})
    groups = feature_cfg.get("feature_groups")
    if groups is None:
        feature_cfg["feature_groups"] = None
    else:
        feature_cfg["feature_groups"] = sorted(set(groups) | set(REQUIRED_EXTRACTOR_GROUPS))
    extractor.begin(config=feature_cfg)

    hf_cfg = dict(config.get("hit_filter", {}) or {})
    hit_filter = stationHitFilter(
        complete_time_check=hf_cfg.get("complete_time_check", True),
        complete_hit_check=hf_cfg.get("complete_hit_check", True),
    )
    hit_filter.begin()

    return extractor, hit_filter


def get_reco3d_config(config: Mapping) -> dict:
    reco_cfg = dict(config.get("reco3d", {}) or {})
    config_file = reco_cfg.pop("config_file", None)
    enabled = reco_cfg.pop("enabled", False)
    if config_file:
        path = Path(os.path.expandvars(str(config_file))).expanduser()
        with path.open("r") as f:
            loaded = yaml.safe_load(f) or {}
        for key, value in list(loaded.items()):
            if isinstance(value, str):
                loaded[key] = os.path.expandvars(value)
        loaded.update(reco_cfg)
        reco_cfg = loaded
    reco_cfg["enabled"] = enabled
    reco_cfg.setdefault("station_id", int(config["station"]))
    reco_cfg.setdefault("channels", list(DEEP_CHS))
    reco_cfg.setdefault("coord_system", "cylindrical")
    reco_cfg.setdefault("validation", True)
    return reco_cfg


def reco3d_module_config(reco_cfg: Mapping) -> dict:
    module_cfg = dict(reco_cfg)
    for key in ("enabled", "config_file", "upsampling_rate_ghz", "preprocessor", "reco_script", "reco_mode"):
        module_cfg.pop(key, None)
    return module_cfg


def update_detector_for_reco_begin(det, reco_cfg: Mapping) -> None:
    """Match the RNOG 3D reco example: reco.begin expects detector time set."""
    if hasattr(det, "get_detector_time"):
        try:
            if det.get_detector_time() is not None:
                return
        except Exception:
            pass
    detector_date = reco_cfg.get("detector_date")
    if detector_date is None or not hasattr(det, "update"):
        return
    if isinstance(detector_date, dt.datetime):
        det_time = detector_date
    else:
        det_time = dt.datetime.fromisoformat(str(detector_date))
    det.update(det_time)


def init_reco3d_tool(config: Mapping, det):
    reco_cfg = get_reco3d_config(config)
    if not reco_cfg.get("enabled", False):
        return None, None

    from NuRadioReco.modules.channelResampler import channelResampler
    from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

    station_id = int(reco_cfg["station_id"])
    reco = InterferometricReco3D()
    update_detector_for_reco_begin(det, reco_cfg)
    reco.begin(station_id, reco3d_module_config(reco_cfg), det)

    resampler = None
    if reco_cfg.get("apply_upsampling", True):
        resampler = channelResampler()
        resampler.begin()
    return reco, resampler


def compute_coherent_snr_deep(station, channels: Iterable[int] = DEEP_CHS) -> float:
    import NuRadioReco.utilities.trace_utilities as trace_utils

    traces = {
        ch_id: np.asarray(station.get_channel(ch_id).get_trace())
        for ch_id in channels
        if station.has_channel(ch_id)
    }
    if not traces:
        return float("nan")

    available = sorted(traces)
    ref_id = available[0]
    ref = traces[ref_id]
    others = [traces[ch_id] for ch_id in available if ch_id != ref_id]
    coherent = trace_utils.get_coherent_sum(others, ref) if others else ref
    noise_rms = trace_utils.get_split_trace_noise_RMS(coherent)
    return float(trace_utils.get_signal_to_noise_ratio(coherent, noise_rms))


def extract_station_traces(station, channels: Iterable[int]) -> Dict[int, np.ndarray]:
    return {
        ch_id: np.asarray(station.get_channel(ch_id).get_trace())
        for ch_id in channels
        if station.has_channel(ch_id)
    }


def coherent_sum_trace(traces: Mapping[int, np.ndarray], channels: Iterable[int]) -> Optional[np.ndarray]:
    import NuRadioReco.utilities.trace_utilities as trace_utils

    available = [ch_id for ch_id in channels if ch_id in traces]
    if not available:
        return None
    ref_id = available[0]
    ref = traces[ref_id]
    others = [traces[ch_id] for ch_id in available if ch_id != ref_id]
    return trace_utils.get_coherent_sum(others, ref) if others else ref


def get_sampling_rate(station, channels: Iterable[int]) -> float:
    for ch_id in channels:
        if station.has_channel(ch_id):
            return float(station.get_channel(ch_id).get_sampling_rate())
    raise ValueError(f"station has none of the requested channels {tuple(channels)}")


def mean_channel_feature(per_ch: Mapping[int, Mapping[str, float]], channels: Iterable[int], name: str) -> float:
    vals = [
        float(per_ch[ch_id][name])
        for ch_id in channels
        if ch_id in per_ch and name in per_ch[ch_id]
    ]
    return float(np.mean(vals)) if vals else float("nan")


def compute_coherent_spectral_pa(station, extractor) -> Dict[str, np.float32]:
    import NuRadioReco.utilities.trace_utilities as trace_utils

    traces = extract_station_traces(station, PA_CHS)
    coherent = coherent_sum_trace(traces, PA_CHS)
    if coherent is None:
        spectral = {}
    else:
        cfg = getattr(extractor, "_config", {})
        spectral = trace_utils.get_spectral_features(
            coherent,
            get_sampling_rate(station, PA_CHS),
            fmin=cfg.get("spectral_fmin"),
            fmax=cfg.get("spectral_fmax"),
            low_band_boundary=cfg.get("spectral_low_band_boundary", 0.1),
        )
    return {
        "coherent_spectral_centroid_pa": np.float32(
            spectral.get("spectral_centroid", np.nan)
        ),
        "coherent_spectral_peak_frequency_pa": np.float32(
            spectral.get("spectral_peak_frequency", np.nan)
        ),
        "coherent_spectral_entropy_pa": np.float32(
            spectral.get("spectral_entropy", np.nan)
        ),
    }


def compute_hit_filter_features(event, station, det, hit_filter) -> Dict[str, np.number]:
    passed = hit_filter.run(event, station, det)
    features: Dict[str, np.number] = {
        "passed_hit_filter": np.int8(1 if passed else 0),
        "n_coincident_pairs_deep": np.int32(-1),
    }
    try:
        in_time_window = hit_filter.is_in_time_window()
        n_pairs_pa = int(sum(in_time_window[0]))
        n_pairs_deep = n_pairs_pa + int(
            sum(in_time_window[grp][0] for grp in range(1, 4))
        )
        features["n_coincident_pairs_deep"] = np.int32(n_pairs_deep)
    except Exception:
        LOG.debug("Could not read stationHitFilter coincidence counters", exc_info=True)
    return features


def get_station_max_corr(station) -> float:
    """Read max correlation if a reco module has already stored it.

    TODO: run/configure ``directionReconstructionDeepCRsearch`` here if the ML
    dataset should contain a real reconstructed max-correlation value.
    """
    try:
        from NuRadioReco.framework.parameters import stationParameters as stnp

        if station.has_parameter(stnp.rec_max_correlation):
            return float(station.get_parameter(stnp.rec_max_correlation))
    except Exception:
        pass
    return float("nan")


def compute_reco3d_features(event, station, det, reco3d, reco_resampler, config: Mapping) -> Dict[str, np.number]:
    out = {
        "surf_corr_zen": np.float32(np.nan),
        "rho": np.float32(np.nan),
        "phi": np.float32(np.nan),
        "z": np.float32(np.nan),
        "max_corr": np.float32(get_station_max_corr(station)),
    }
    if reco3d is None:
        return out

    try:
        reco_cfg = get_reco3d_config(config)

        if reco_resampler is not None:
            from NuRadioReco.utilities import units

            rate_ghz = float(reco_cfg.get("upsampling_rate_ghz", 10.0))
            reco_resampler.run(event, station, det, sampling_rate=rate_ghz * units.GHz)

        result = reco3d.run(event, station, det, reco3d_module_config(reco_cfg))
        out.update({
            "surf_corr_zen": np.float32(result.get("surf_corr_zen", np.nan)),
            "rho": np.float32(result.get("rho", np.nan)),
            "phi": np.float32(result.get("phi", np.nan)),
            "z": np.float32(result.get("z", np.nan)),
            "max_corr": np.float32(result.get("max_corr", out["max_corr"])),
        })
    except Exception:
        LOG.warning("3D interferometric reco failed for event", exc_info=True)
    return out


def compute_features(event, station, det, extractor, hit_filter, reco3d, reco_resampler, config) -> Dict[str, np.number]:
    per_ch = extractor.run(event, station, det, channel_ids=DEEP_CHS)
    snrs = [features["snr"] for features in per_ch.values() if "snr" in features]
    hit_features = compute_hit_filter_features(event, station, det, hit_filter)
    pa_amp = mean_channel_feature(per_ch, PA_CHS, "max_amplitude_envelope")
    hpol_amp = mean_channel_feature(per_ch, HPOL_CHS, "max_amplitude_envelope")
    pa_over_hpol = (
        float(pa_amp / hpol_amp)
        if np.isfinite(pa_amp) and np.isfinite(hpol_amp) and hpol_amp != 0
        else float("nan")
    )

    features = {
        "snr_avg_deep": np.float32(np.mean(snrs) if snrs else np.nan),
        "max_amplitude_avg_pa_over_hpol": np.float32(pa_over_hpol),
        "impulse_corr_bipolar_wide_avg_pa": np.float32(
            mean_channel_feature(per_ch, PA_CHS, "impulse_corr_bipolar_wide")
        ),
        "impulse_corr_bipolar_avg_pa": np.float32(
            mean_channel_feature(per_ch, PA_CHS, "impulse_corr_bipolar")
        ),
    }
    features.update(compute_coherent_spectral_pa(station, extractor))
    features["coherent_snr_deep"] = np.float32(compute_coherent_snr_deep(station))
    features.update(compute_reco3d_features(event, station, det, reco3d, reco_resampler, config))
    features.update(hit_features)
    return features


def read_calibrate_events(
    input_files: List[str],
    station_id: int,
    det,
    config: Mapping,
    sim_truth: Optional[Mapping[str, np.ndarray]] = None,
) -> Iterator[Dict[str, object]]:
    """Yield processed ML records for one run."""
    provider = init_provider(input_files, det, config)
    extractor, hit_filter = init_feature_tools(config)
    reco3d = None
    reco_resampler = None
    reco_init_attempted = not get_reco3d_config(config).get("enabled", False)
    try:
        for source_event_index, event in enumerate(provider.run()):
            try:
                maybe_process_event(event, det, config)
                station = event.get_station(station_id)
                if not reco_init_attempted:
                    reco_init_attempted = True
                    try:
                        reco3d, reco_resampler = init_reco3d_tool(config, det)
                    except Exception:
                        LOG.warning(
                            "3D interferometric reco initialization failed; "
                            "continuing with reco features set to NaN.",
                            exc_info=True,
                        )
                maybe_resample_output(event, station, det, config)
                trace_start_time, trace_delta_t = extract_waveform_timing(station)
                truth_row = sim_truth_for_event(sim_truth or {}, source_event_index)
                yield {
                    "waveforms": extract_waveforms(station),
                    "trace_start_time": trace_start_time,
                    "trace_delta_t": trace_delta_t,
                    "run": np.int32(event.get_run_number()),
                    "event_number": np.int32(event.get_id()),
                    "trigger_type": get_trigger_code(station),
                    "timestamp": get_timestamp(station),
                    "sim_truth": truth_row,
                    "weight": (
                        truth_row.get("weight") if config.get("use_sim_weights", True) else None
                    ),
                    "features": compute_features(
                        event, station, det, extractor, hit_filter, reco3d, reco_resampler, config
                    ),
                }
            except Exception as exc:
                LOG.warning(
                    "Skipping run %s event %s after processing failure: %s",
                    getattr(event, "get_run_number", lambda: "<unknown>")(),
                    getattr(event, "get_id", lambda: "<unknown>")(),
                    exc,
                )
                yield {"_skip": True}
    finally:
        provider.end()
        hit_filter.end()
        if reco3d is not None:
            reco3d.end()
        if reco_resampler is not None and hasattr(reco_resampler, "end"):
            reco_resampler.end()


def create_or_resize_dataset(h5: h5py.File, name: str, shape_tail, dtype, chunks, compression):
    return h5.create_dataset(
        name,
        shape=(0, *shape_tail),
        maxshape=(None, *shape_tail),
        dtype=dtype,
        chunks=chunks,
        compression=compression,
    )


def append_one(ds, value) -> None:
    n = ds.shape[0]
    ds.resize((n + 1, *ds.shape[1:]))
    ds[n] = value


def create_dynamic_dataset(group, name: str, value, chunks_first_dim: int, compression):
    arr = np.asarray(value)
    shape_tail = arr.shape
    chunks = (chunks_first_dim, *shape_tail)
    return create_or_resize_dataset(group, name, shape_tail, arr.dtype, chunks, compression)


def preprocessing_version(config: Mapping) -> str:
    explicit = config.get("preprocessing_version")
    if explicit:
        return str(explicit)
    root = config.get("nuradio_path") or config.get("nuradiomc_path")
    if not root:
        return "unknown"
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def write_hdf5(
    records: Iterator[Dict[str, object]],
    output_path: Path,
    attrs: Mapping[str, object],
    label: int = 0,
    source_type: int = 0,
    event_weight: float = 1.0,
    compression: str = "gzip",
    waveform_chunk_events: int = 64,
) -> Tuple[int, int, int]:
    """Write one run HDF5 file incrementally."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    read_count = 0
    written_count = 0
    skipped_count = 0

    with h5py.File(tmp_path, "w") as h5:
        wf_chunks = (waveform_chunk_events, N_CHANNELS, N_SAMPLES)
        datasets = {
            "waveforms": create_or_resize_dataset(
                h5, "waveforms", (N_CHANNELS, N_SAMPLES), "f4", wf_chunks, compression
            ),
            "trace_start_time": create_or_resize_dataset(
                h5, "trace_start_time", (N_CHANNELS,), "f8",
                (waveform_chunk_events, N_CHANNELS), compression
            ),
            "trace_delta_t": create_or_resize_dataset(
                h5, "trace_delta_t", (N_CHANNELS,), "f8",
                (waveform_chunk_events, N_CHANNELS), compression
            ),
            "run": create_or_resize_dataset(
                h5, "run", (), "i4", (waveform_chunk_events,), compression
            ),
            "event_number": create_or_resize_dataset(
                h5, "event_number", (), "i4", (waveform_chunk_events,), compression
            ),
            "trigger_type": create_or_resize_dataset(
                h5, "trigger_type", (), "i1", (waveform_chunk_events,), compression
            ),
            "timestamp": create_or_resize_dataset(
                h5, "timestamp", (), "f8", (waveform_chunk_events,), compression
            ),
            "label": create_or_resize_dataset(
                h5, "label", (), "i1", (waveform_chunk_events,), compression
            ),
            "source_type": create_or_resize_dataset(
                h5, "source_type", (), "i1", (waveform_chunk_events,), compression
            ),
            "weight": create_or_resize_dataset(
                h5, "weight", (), "f4", (waveform_chunk_events,), compression
            ),
        }
        sim_group = h5.create_group("sim")
        sim_datasets = {}
        feat_group = h5.create_group("features")
        feature_datasets = {
            name: create_or_resize_dataset(
                feat_group, name, (), dtype, (waveform_chunk_events,), compression
            )
            for name, dtype in FEATURE_DTYPES.items()
        }
        feature_datasets["max_corr"].attrs["sentinel"] = "NaN means not computed"
        feature_datasets["max_corr"].attrs["todo"] = (
            "Configure/run directionReconstructionDeepCRsearch before reading "
            "stationParameters.rec_max_correlation."
        )
        datasets["trigger_type"].attrs["sentinel"] = "-1 means unavailable"
        datasets["trigger_type"].attrs["codes"] = str(TRIGGER_CODE)
        datasets["timestamp"].attrs["sentinel"] = "NaN means unavailable"
        datasets["label"].attrs["description"] = "ML target label; 0=noise/background, 1=signal."
        datasets["source_type"].attrs["codes"] = str(SOURCE_TYPE_CODE)
        datasets["weight"].attrs["description"] = "Per-event training/evaluation weight."
        datasets["trace_start_time"].attrs["description"] = (
            "Time of the first waveform sample for each channel."
        )
        datasets["trace_start_time"].attrs["units"] = (
            "NuRadioReco time units from channel.get_trace_start_time()"
        )
        datasets["trace_delta_t"].attrs["description"] = (
            "Sample spacing for each channel, computed as 1 / channel.get_sampling_rate()."
        )
        datasets["trace_delta_t"].attrs["units"] = (
            "NuRadioReco time units from channel.get_sampling_rate() reciprocal"
        )

        for key, value in attrs.items():
            h5.attrs[key] = value

        for record in records:
            read_count += 1
            if record.get("_skip"):
                skipped_count += 1
                continue
            try:
                waveforms = record["waveforms"]
                if waveforms.shape != DEFAULT_WAVEFORM_SHAPE:
                    raise ValueError(
                        f"waveform shape {waveforms.shape}, expected {DEFAULT_WAVEFORM_SHAPE}"
                    )
                append_one(datasets["waveforms"], waveforms)
                append_one(datasets["trace_start_time"], record["trace_start_time"])
                append_one(datasets["trace_delta_t"], record["trace_delta_t"])
                append_one(datasets["run"], record["run"])
                append_one(datasets["event_number"], record["event_number"])
                append_one(datasets["trigger_type"], record["trigger_type"])
                timestamp = record["timestamp"]
                append_one(datasets["timestamp"], np.nan if timestamp is None else timestamp)
                append_one(datasets["label"], np.int8(label))
                append_one(datasets["source_type"], np.int8(source_type))
                weight = record.get("weight")
                append_one(
                    datasets["weight"],
                    np.float32(event_weight if weight is None else weight),
                )
                for name, value in record.get("sim_truth", {}).items():
                    if name == "weight":
                        continue
                    if name not in sim_datasets:
                        sim_datasets[name] = create_dynamic_dataset(
                            sim_group, name, value, waveform_chunk_events, compression
                        )
                    append_one(sim_datasets[name], value)
                for name in FEATURE_DTYPES:
                    append_one(feature_datasets[name], record["features"][name])
                written_count += 1
            except Exception as exc:
                skipped_count += 1
                LOG.warning("Skipping event after processing failure: %s", exc)

    tmp_path.replace(output_path)
    return read_count, written_count, skipped_count


def validate_hdf5(path: Path) -> None:
    with h5py.File(path, "r") as h5:
        n = h5["waveforms"].shape[0]
        if h5["waveforms"].shape[1:] != DEFAULT_WAVEFORM_SHAPE:
            raise ValueError(f"{path}: bad waveforms shape {h5['waveforms'].shape}")
        for name in ("trace_start_time", "trace_delta_t"):
            if h5[name].shape != (n, N_CHANNELS):
                raise ValueError(
                    f"{path}: /{name} shape {h5[name].shape} != {(n, N_CHANNELS)}"
                )
        for name in (
            "run", "event_number", "trigger_type", "timestamp",
            "label", "source_type", "weight",
        ):
            if h5[name].shape != (n,):
                raise ValueError(f"{path}: /{name} shape {h5[name].shape} != {(n,)}")
        for name in FEATURE_DTYPES:
            if h5[f"features/{name}"].shape != (n,):
                raise ValueError(
                    f"{path}: /features/{name} shape {h5[f'features/{name}'].shape} != {(n,)}"
                )
        if "sim" in h5:
            for name, ds in h5["sim"].items():
                if ds.shape[0] != n:
                    raise ValueError(f"{path}: /sim/{name} shape {ds.shape} has {ds.shape[0]} rows != {n}")


def run_range(config: Mapping) -> List[int]:
    if "runs" in config:
        return [int(r) for r in config["runs"]]
    if "run" in config:
        return [int(config["run"])]
    return list(range(int(config["run_start"]), int(config["run_end"]) + 1))


def build_run(station: int, run: int, config: Mapping) -> None:
    input_dir = Path(config["input_dir"]).expanduser()
    output_dir = Path(config["output_dir"]).expanduser()
    input_files = discover_input_files(input_dir, station, run)
    det = init_detector(config)

    output_path = output_dir / f"station{station}_run{run}.h5"
    attrs = {
        "station": station,
        "run": run,
        "source_root_path": ",".join(input_files),
        "preprocessing_version": preprocessing_version(config),
        "waveform_units": config.get("waveform_units", "NuRadioReco voltage units"),
        "creation_time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label": int(config.get("label", 0)),
        "source_type": int(get_source_type_code(config)),
        "dataset_year": config.get("year", ""),
    }
    records = read_calibrate_events(input_files, station, det, config)
    read_count, written_count, skipped_count = write_hdf5(
        records,
        output_path,
        attrs,
        label=int(config.get("label", 0)),
        source_type=int(get_source_type_code(config)),
        event_weight=float(config.get("event_weight", 1.0)),
        compression=config.get("compression", "gzip"),
        waveform_chunk_events=int(config.get("waveform_chunk_events", 64)),
    )
    validate_hdf5(output_path)
    print(
        f"Run {run}: events read={read_count}, written={written_count}, "
        f"skipped={skipped_count}, output={output_path}",
        flush=True,
    )


def configured_input_files(config: Mapping) -> Optional[List[str]]:
    patterns = config.get("input_files")
    if patterns is None:
        patterns = config.get("input_glob")
    if patterns is None:
        return None
    if isinstance(patterns, str):
        patterns = [patterns]
    return expand_input_files(patterns)


def build_input_file(station: int, input_file: str, config: Mapping) -> None:
    output_dir = Path(config["output_dir"]).expanduser()
    det = init_detector(config)
    input_path = Path(input_file)
    output_prefix = config.get("output_prefix", f"station{station}")
    output_path = output_dir / f"{output_prefix}_{input_path.stem}.h5"
    truth_path = paired_sim_truth_path(input_file)
    sim_truth = load_sim_truth(truth_path, station)
    attrs = {
        "station": station,
        "run": -1,
        "source_root_path": input_file,
        "sim_truth_path": "" if truth_path is None else str(truth_path),
        "preprocessing_version": preprocessing_version(config),
        "waveform_units": config.get("waveform_units", "NuRadioReco voltage units"),
        "creation_time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label": int(config.get("label", 1)),
        "source_type": int(get_source_type_code(config)),
        "dataset_year": config.get("year", ""),
    }
    records = read_calibrate_events([input_file], station, det, config, sim_truth=sim_truth)
    read_count, written_count, skipped_count = write_hdf5(
        records,
        output_path,
        attrs,
        label=int(config.get("label", 1)),
        source_type=int(get_source_type_code(config)),
        event_weight=float(config.get("event_weight", 1.0)),
        compression=config.get("compression", "gzip"),
        waveform_chunk_events=int(config.get("waveform_chunk_events", 64)),
    )
    validate_hdf5(output_path)
    print(
        f"{input_path.name}: events read={read_count}, written={written_count}, "
        f"skipped={skipped_count}, output={output_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="YAML config file")
    parser.add_argument("--run", type=int, help="Override config with a single data run")
    parser.add_argument(
        "--input-file",
        action="append",
        help="Override config with one input file. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, help="Override config output_dir")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    config = load_config(args.config)
    if args.run is not None:
        config["run"] = args.run
        config.pop("runs", None)
        config.pop("run_start", None)
        config.pop("run_end", None)
        config.pop("input_files", None)
        config.pop("input_glob", None)
    if args.input_file:
        config["input_files"] = args.input_file
        config.pop("run", None)
        config.pop("runs", None)
        config.pop("run_start", None)
        config.pop("run_end", None)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    configure_nuradio_path(config)

    station = int(config["station"])
    input_files = configured_input_files(config)
    successes = 0
    failures = 0
    if input_files is not None:
        for input_file in input_files:
            try:
                build_input_file(station, input_file, config)
                successes += 1
            except Exception as exc:
                failures += 1
                if config.get("continue_on_run_error", True):
                    LOG.warning("Skipping input %s after failure: %s", input_file, exc)
                    continue
                raise
    else:
        for run in run_range(config):
            try:
                build_run(station, run, config)
                successes += 1
            except Exception as exc:
                failures += 1
                if config.get("continue_on_run_error", True):
                    LOG.warning("Skipping run %s after failure: %s", run, exc)
                    continue
                raise
    if failures and successes == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
