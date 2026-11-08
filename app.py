# -*- coding: utf-8 -*-
"""
Streamlit web app: R-L phase difference, delay, level and 2nd harmonic vs time
or groove radius, with tracking-error curve fit (Sperling / Clearaudio records).

Based on ANYSIGNAL_-phasediff_2harm_sweep__R15.py by ersol (Balle-Clorin).
The DSP is the R15 code unchanged; only input/output was adapted for the web:
settings come from the sidebar instead of the script header, the file is
uploaded, console messages are shown on the page, and the CSV is downloaded.

License: PolyForm Noncommercial 1.0.0
"""
import gc
import io
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import scipy.signal as sig
import scipy.optimize as opt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import pandas as pd
import streamlit as st

# ---------------------------
# App identity (shown in the result header)
# ---------------------------
SCRIPT_NAME = "ANYSIGNAL_-phasediff_2harm_sweep"
SCRIPT_REV = "R15-web2"

# ---------------------------
# Fixed constants (as in R15)
# ---------------------------
EPS = 1e-20
MIN_FUND_AMPLITUDE = 1e-6
MIN_RATIO_DB = -80.0
WINDOW = "hann"
BASE_FONT = 13
BG = "#1e1e1e"
FG = "#e6e6e6"
FIT_COLOR = "#ffffff"       # was black ('k--') in R15; white on the dark theme

RECORD_PROFILES = {
    "sperling": {
        "label": "Sperling",
        "sides": {   # 1 kHz band, outer -> inner; constant pitch => radius linear in time
            "33": {"rpm": 100.0 / 3.0, "duration_s": 22 * 60 + 17, "r_start_mm": 146.05, "r_end_mm": 57.5},
            "45": {"rpm": 45.0,        "duration_s": 16 * 60 + 5,  "r_start_mm": 146.05, "r_end_mm": 57.5},
        },
        "markers_mm": (120.9, 66.0),
        "tone_hz": 1000.0,
    },
    "clearaudio": {
        "label": "clearaudio",
        "tone_hz": 3150.0,
        "sides": {
            "33": {"rpm": 100.0 / 3.0, "duration_s": 14 * 60 + 58, "r_start_mm": 127, "r_end_mm": 69.0},
        },
        "markers_mm": (),
    },
}

DEFAULTS = dict(
    MODE="peak", FREQ_CENTER=1000.0, FREQ_TOL_HZ=20.0,
    SWEEP_BAND=(20.0, 48000.0), FREQ_BAND=(900.0, 1100.0),
    NFFT=32768, HOP=2048, PHASE_UNIT="degrees", UNWRAP_PHASE=True, SMOOTH_MS=18000.0,
    TITLE_TEXT="", Y_AXIS_LIMITS=None,
    RECORD_PROFILE="sperling", RECORD_SIDE="auto", RADIUS_ALIGN="both", RADIUS_T_OFFSET_S=0.0,
    RADIUS_MAX_SPAN_DEV=0.05, RADIUS_DETECT_DROP_DB=10.0, RADIUS_DETECT_HOLD_S=1.0,
    RADIUS_ENV_BLOCK_MS=10.0, RADIUS_INVERT_X=False, RADIUS_XLIM_MM=(55.0, 150.0),
    HARM_RADIUS_NORM=True, HARM_NORM_REF_MM=None, HARM_NORM_SHOW_RAW=True,
    SHOW_PAGE2=True,
    PHASE_FIT_TRACKING_ERROR=True, PHASE_FIT_MODEL="geometry",
    PHASE_FIT_N1_GUESS_MM=66.0, PHASE_FIT_N2_GUESS_MM=121.0,
    PHASE_FIT_PIVOT_SPINDLE_MM=215.35, PHASE_FIT_EFF_LENGTH_MM=233.0, PHASE_FIT_OVERHANG_GUESS_MM=17.8,
    PHASE_FIT_OFFSET_ANGLE_GUESS_DEG=23.6, PHASE_FIT_TRIM_S=10.0,
    PHASE_FIT_ALPHA_KNOWN=False, PHASE_FIT_SHOW_ZENITH_CURVE=True, PHASE_FIT_SHOW_OPTIMAL=True, PHASE_FIT_LOFGREN_R1_MM=60.3, PHASE_FIT_LOFGREN_R2_MM=146.05,
    PHASE_FIT_OFFSET_DEG=0.0,
    # Angular error: phase is not proportional to tracking angle error. At a given angular error the phase grows
    # as the groove gets slower and the recorded wavelength shorter, ~ (r_ref/r)^p with p = 1 for a response
    # proportional to 1/groove speed. ANG_NORM converts phase to angular error (units: degrees of phase this angular
    # error would give at r_ref) and the geometry fits use the same response. ANG_NORM=False gives R15 behaviour.
    ANG_NORM=True, ANG_REF_MM=None, ANG_EXPONENT=1.0,
)


# ---------------------------
# Utilities (R15)
# ---------------------------
def smooth_signal(x, sr_frames, smooth_ms):
    if smooth_ms <= 0:
        return x
    kernel_len = max(1, int(round(smooth_ms * sr_frames / 1000.0)))
    kernel = np.ones(kernel_len) / kernel_len
    return np.convolve(x, kernel, mode='same')


def detect_tone_span(path, block_ms, drop_db, hold_s):
    """Return (t_on, t_off, ref_db) of the sustained tone, or None (R15, unchanged)."""
    with sf.SoundFile(path, 'r') as f:
        sr_ = f.samplerate
        blk = max(1, int(round(sr_ * block_ms / 1000.0)))
        rms_parts = []
        while True:
            b = f.read(frames=blk * 1000, dtype='float32', always_2d=True)
            nb = len(b) // blk
            if nb == 0:
                break
            b = b[:nb * blk].reshape(nb, blk, b.shape[1])
            rms_parts.append(np.sqrt(np.mean(b.astype(np.float64) ** 2, axis=1)).max(axis=1))
            del b
    if not rms_parts:
        return None
    env_db = 20.0 * np.log10(np.maximum(np.concatenate(rms_parts), EPS))
    ref_db = np.percentile(env_db, 90)
    above = env_db > (ref_db - drop_db)
    K = max(1, int(round(hold_s * 1000.0 / block_ms)))
    if above.size < K:
        return None
    cs = np.concatenate(([0], np.cumsum(above)))
    run_ok = (cs[K:] - cs[:-K]) == K
    idx = np.flatnonzero(run_ok)
    if idx.size == 0:
        return None
    blk_s = blk / sr_
    return idx[0] * blk_s, (idx[-1] + K) * blk_s, ref_db


# ---------------------------
# Analysis (R15 processing, wrapped in a function)
# ---------------------------
def run_analysis(path, display_name, C, progress=None):
    log = []          # (level, text): level "info" or "warn"

    def say(txt, level="info"):
        log.append((level, txt))

    MODE = C["MODE"]
    # tone frequency: explicit override > record profile > default
    _prof_hz = RECORD_PROFILES[C["RECORD_PROFILE"]].get("tone_hz") if C["RECORD_PROFILE"] else None
    if C["FREQ_CENTER"] and C["FREQ_CENTER"] > 0:
        C = dict(C, FREQ_CENTER=float(C["FREQ_CENTER"]))
        _hz_src = "manual"
    else:
        C = dict(C, FREQ_CENTER=float(_prof_hz or DEFAULTS["FREQ_CENTER"]))
        _hz_src = "record profile" if _prof_hz else "default"
    NFFT, HOP = int(C["NFFT"]), int(C["HOP"])
    SMOOTH_MS = C["SMOOTH_MS"]
    PHASE_UNIT = C["PHASE_UNIT"]

    sf_desc = sf.info(path)
    if sf_desc.channels < 2:
        raise ValueError("Audio must be stereo (2 channels).")

    sf_file = sf.SoundFile(path, 'r')
    sr = sf_file.samplerate
    n_channels = sf_file.channels
    total_frames = len(sf_file)
    say(f"Opened: {display_name}  sr={sr} channels={n_channels} frames={total_frames} subtype={sf_desc.subtype}")
    if MODE == "peak":
        say(f"Peak mode: tracking tone at {C['FREQ_CENTER']:.0f} Hz ± {C['FREQ_TOL_HZ']:.0f} Hz ({_hz_src})")

    if HOP >= NFFT:
        raise ValueError("HOP must be smaller than NFFT.")

    win = sig.get_window(WINDOW, NFFT, fftbins=True)
    freqs = np.fft.rfftfreq(NFFT, d=1.0 / sr)

    if MODE == "peak":
        mask_peak = (freqs >= (C["FREQ_CENTER"] - C["FREQ_TOL_HZ"])) & (freqs <= (C["FREQ_CENTER"] + C["FREQ_TOL_HZ"]))
        if not np.any(mask_peak):
            raise ValueError("Search window out of FFT band. Increase NFFT or reduce the tone frequency.")
        peak_idxs = np.where(mask_peak)[0]
    else:
        peak_idxs = None

    FREQ_BAND = C["FREQ_BAND"]
    if MODE == "sweep":
        search_band = C["SWEEP_BAND"] if C["SWEEP_BAND"] is not None else FREQ_BAND
        if search_band is None:
            search_band = (freqs[1], freqs[-1])
        fmin_s, fmax_s = search_band
        sweep_mask = (freqs >= fmin_s) & (freqs <= fmax_s)
        if not np.any(sweep_mask):
            raise ValueError("Sweep band outside FFT range; adjust NFFT or band.")
        sweep_idxs = np.where(sweep_mask)[0]
    else:
        sweep_idxs = None

    if FREQ_BAND is not None:
        fmin, fmax = FREQ_BAND
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(band_mask):
            raise ValueError("Broadband band outside FFT range; increase NFFT or change band.")
    else:
        band_mask = np.ones_like(freqs, dtype=bool)

    buf = np.zeros((NFFT, n_channels), dtype='float32')

    # global time-domain peak for dBFS normalization
    peak_abs = 0.0
    sf_file.seek(0)
    while True:
        block = sf_file.read(frames=262144, dtype='float32', always_2d=True)
        if len(block) == 0:
            break
        peak_abs = max(peak_abs, float(np.max(np.abs(block))))
    del block
    if peak_abs < EPS:
        raise ValueError("Input is essentially silent.")
    sf_file.seek(0)

    times, phase_vals, coherence_vals = [], [], []
    level_L, level_R, tracked_freqs = [], [], []
    harmonic_linear_L, harmonic_db_L, harmonic_valid_L = [], [], []
    harmonic_linear_R, harmonic_db_R, harmonic_valid_R = [], [], []

    def harm_append(XL, XR, idx, f0):
        f_h = 2.0 * f0
        if f_h <= freqs[-1]:
            idx_h = int(np.argmin(np.abs(freqs - f_h)))
            amp_f_L = np.abs(XL[idx]); amp_h_L = np.abs(XL[idx_h])
            valid_L = (amp_f_L > MIN_FUND_AMPLITUDE)
            if valid_L:
                lin_L = amp_h_L / (amp_f_L + EPS); db_L = 20.0 * math.log10(max(lin_L, EPS))
            else:
                lin_L = 0.0; db_L = MIN_RATIO_DB; valid_L = False
            harmonic_linear_L.append(lin_L); harmonic_db_L.append(db_L); harmonic_valid_L.append(bool(valid_L))
            amp_f_R = np.abs(XR[idx]); amp_h_R = np.abs(XR[idx_h])
            valid_R = (amp_f_R > MIN_FUND_AMPLITUDE)
            if valid_R:
                lin_R = amp_h_R / (amp_f_R + EPS); db_R = 20.0 * math.log10(max(lin_R, EPS))
            else:
                lin_R = 0.0; db_R = MIN_RATIO_DB; valid_R = False
            harmonic_linear_R.append(lin_R); harmonic_db_R.append(db_R); harmonic_valid_R.append(bool(valid_R))
        else:
            harmonic_linear_L.append(np.nan); harmonic_db_L.append(np.nan); harmonic_valid_L.append(False)
            harmonic_linear_R.append(np.nan); harmonic_db_R.append(np.nan); harmonic_valid_R.append(False)

    first_block = sf_file.read(frames=NFFT, dtype='float32', always_2d=True)
    if first_block.shape[0] < NFFT:
        pad = np.zeros((NFFT - first_block.shape[0], n_channels), dtype='float32')
        buf = np.vstack((first_block, pad))
    else:
        buf[:, :] = first_block[:, :]
    del first_block

    frame_index = 0
    read_eof = (len(sf_file) <= NFFT)
    n_frames_est = max(1, (total_frames - NFFT) // HOP + 2)

    while True:
        left_frame = buf[:, 0]
        right_frame = buf[:, 1]
        XL = np.fft.rfft(left_frame * win)
        XR = np.fft.rfft(right_frame * win)
        Cx = XR * np.conj(XL)    # phase = phase_R - phase_L

        if MODE == "broadband":
            C_band = Cx[band_mask]
            w = np.abs(C_band)
            w_sum = np.sum(w)
            if w_sum == 0:
                phase_avg, coherence_like, f_avg = 0.0, 0.0, np.nan
            else:
                numerator = np.sum(w * C_band)
                phase_avg = np.angle(numerator / (w_sum + EPS))
                coherence_like = np.abs(numerator) / (w_sum + EPS)
                f_avg = np.sum(w * freqs[band_mask]) / (w_sum + EPS)
            phase_vals.append(phase_avg); coherence_vals.append(coherence_like); tracked_freqs.append(f_avg)
            harmonic_linear_L.append(np.nan); harmonic_db_L.append(np.nan); harmonic_valid_L.append(False)
            harmonic_linear_R.append(np.nan); harmonic_db_R.append(np.nan); harmonic_valid_R.append(False)
        else:
            idxs = peak_idxs if MODE == "peak" else sweep_idxs
            mags_search = np.abs(XL[idxs])
            idx = idxs[int(np.argmax(mags_search))]
            f0 = freqs[idx]
            tracked_freqs.append(f0)
            phase_vals.append(np.angle(XR[idx]) - np.angle(XL[idx]))
            coherence_vals.append(np.abs(Cx[idx]) / (np.max(mags_search) + EPS))
            harm_append(XL, XR, idx, f0)

        ampL = np.sqrt(np.mean(left_frame ** 2))
        ampR = np.sqrt(np.mean(right_frame ** 2))
        level_L.append(20.0 * math.log10(max(ampL / peak_abs, EPS)))
        level_R.append(20.0 * math.log10(max(ampR / peak_abs, EPS)))

        times.append((frame_index * HOP) / sr)

        if progress is not None and frame_index % 200 == 0:
            progress(min(1.0, frame_index / n_frames_est))

        if read_eof:
            break
        next_block = sf_file.read(frames=HOP, dtype='float32', always_2d=True)
        if next_block.shape[0] == 0:
            read_eof = True
            buf[:-HOP, :] = buf[HOP:, :]
            buf[-HOP:, :] = 0.0
        else:
            n_read = next_block.shape[0]
            buf[:-HOP, :] = buf[HOP:, :]
            if n_read < HOP:
                buf[-HOP:-HOP + n_read, :] = next_block
                buf[-HOP + n_read:, :] = 0.0
                read_eof = True
            else:
                buf[-HOP:, :] = next_block[:HOP, :]
        frame_index += 1

    sf_file.close()
    del buf
    if progress is not None:
        progress(1.0)

    times = np.array(times)
    phase_vals = np.array(phase_vals)
    coherence_vals = np.array(coherence_vals)
    level_L = np.array(level_L); level_R = np.array(level_R)
    tracked_freqs = np.array(tracked_freqs)
    harmonic_linear_L = np.array(harmonic_linear_L); harmonic_db_L = np.array(harmonic_db_L)
    harmonic_valid_L = np.array(harmonic_valid_L, dtype=bool)
    harmonic_linear_R = np.array(harmonic_linear_R); harmonic_db_R = np.array(harmonic_db_R)
    harmonic_valid_R = np.array(harmonic_valid_R, dtype=bool)

    # ---------------- time -> groove radius ----------------
    duration_s = total_frames / sr
    radius_mm = np.full(len(times), np.nan)
    radius_note = ""
    RECORD_PROFILE = C["RECORD_PROFILE"]
    RECORD_SIDE = C["RECORD_SIDE"]
    RADIUS_ALIGN = C["RADIUS_ALIGN"]
    RADIUS_MODE = RECORD_PROFILE is not None
    prof_label, sp_side, prof_markers = None, None, ()
    sp_r0 = sp_r1 = None
    span = None

    if RADIUS_MODE:
        prof = RECORD_PROFILES[RECORD_PROFILE]
        prof_label = prof.get("label", RECORD_PROFILE)
        prof_sides = prof["sides"]
        prof_markers = prof.get("markers_mm", ())
        if RADIUS_ALIGN in ("both", "onset") or RECORD_SIDE == "auto":
            span = detect_tone_span(path, C["RADIUS_ENV_BLOCK_MS"], C["RADIUS_DETECT_DROP_DB"], C["RADIUS_DETECT_HOLD_S"])
        if RECORD_SIDE == "auto":
            ref_len = (span[1] - span[0]) if span is not None else duration_s
            sp_side = min(prof_sides, key=lambda k: abs(prof_sides[k]["duration_s"] - ref_len))
            src_txt = "tone length" if span is not None else "file duration"
            say(f"{prof_label}: auto side from {src_txt} {ref_len:.1f} s -> side {sp_side}")
            if span is not None:
                dev = (ref_len - prof_sides[sp_side]["duration_s"]) / prof_sides[sp_side]["duration_s"]
                if abs(dev) > C["RADIUS_MAX_SPAN_DEV"]:
                    say(f"{prof_label}: tone length {ref_len:.1f} s matches no side of profile "
                        f"'{RECORD_PROFILE}' (closest off by {dev*100:+.1f} %). Wrong profile or partial "
                        f"recording? Plotting vs TIME. Choose the record side explicitly to force radius.", "warn")
                    RADIUS_MODE = False
        elif RECORD_SIDE in prof_sides:
            sp_side = RECORD_SIDE
        else:
            raise ValueError(f"Record side must be 'auto' or one of {list(prof_sides)} for this profile.")

    if RADIUS_MODE:
        sp = prof_sides[sp_side]
        sp_T_nom = float(sp["duration_s"])
        sp_r0, sp_r1 = float(sp["r_start_mm"]), float(sp["r_end_mm"])
        sp_pitch_um = abs(sp_r0 - sp_r1) * 1000.0 / (sp_T_nom * sp["rpm"] / 60.0)
        sp_align = RADIUS_ALIGN
        sp_t0 = C["RADIUS_T_OFFSET_S"]
        sp_T = sp_T_nom
        if sp_align in ("both", "onset"):
            if span is None:
                say(f"{prof_label}: tone not detected, using the manual start offset.", "warn")
                sp_align = "manual"
            else:
                t_on, t_off, ref_db = span
                sp_t0 = t_on
                say(f"{prof_label}: tone detected {t_on:.2f} s -> {t_off:.2f} s "
                    f"(span {t_off - t_on:.2f} s, nominal {sp_T_nom:.0f} s, ref {ref_db:.1f} dBFS)")
                if t_on < 0.5:
                    say(f"{prof_label}: tone present from the very start of the file. The recording may "
                        "have started mid-band; radius assumes band start here.", "warn")
                if sp_align == "both":
                    dev = (t_off - t_on - sp_T_nom) / sp_T_nom
                    if abs(dev) > C["RADIUS_MAX_SPAN_DEV"]:
                        say(f"{prof_label}: detected span deviates {dev*100:+.1f} % from nominal "
                            f"(partial recording?). Using onset + nominal duration.", "warn")
                        sp_align = "onset"
                    else:
                        sp_T = t_off - t_on
        speed_dev_pct = (sp_T_nom / sp_T - 1.0) * 100.0
        t_center = times + 0.5 * NFFT / sr
        radius_mm = sp_r0 + (sp_r1 - sp_r0) * (t_center - sp_t0) / sp_T
        radius_note = (f"{prof_label} side {sp_side} ({sp['rpm']:.2f} rpm): tone {sp_t0:.2f}-{sp_t0 + sp_T:.2f} s -> "
                       f"{sp_r0:.2f}-{sp_r1:.2f} mm (align={sp_align}, pitch {sp_pitch_um:.1f} µm/rev")
        radius_note += f", speed {speed_dev_pct:+.2f} %)" if sp_align == "both" else ")"
        say(radius_note)

    if RADIUS_MODE:
        _band = (radius_mm >= min(sp_r0, sp_r1)) & (radius_mm <= max(sp_r0, sp_r1))
        x_plot = np.where(_band, radius_mm, np.nan)
    else:
        x_plot = times

    # ---------------- phase unwrap / units / smoothing ----------------
    if C["UNWRAP_PHASE"]:
        phase_vals = np.unwrap(phase_vals)
        if RADIUS_MODE:
            _in_band = (radius_mm >= min(sp_r0, sp_r1)) & (radius_mm <= max(sp_r0, sp_r1))
            if _in_band.any():
                _k = np.round(np.median(phase_vals[_in_band]) / (2 * np.pi))
                phase_vals = phase_vals - 2 * np.pi * _k
    phase_plot = np.degrees(phase_vals) if PHASE_UNIT == "degrees" else phase_vals
    sr_frames = 1.0 / (times[1] - times[0]) if len(times) > 1 else 1.0
    phase_smooth = smooth_signal(phase_plot, sr_frames, SMOOTH_MS)

    # ---------------- tracking-error fit ----------------
    phase_fit_active = False
    phase_fit_curve = None
    phase_fit_note = ""
    phase_fit_nulls = None
    phase_opt_curve = None
    phase_zen_curve = None     # optimal (theory) curve shifted by the constant zenith angular offset
    ang_zen_curve = None
    zen_used = None            # (zenith offset, k, fixed?) when the angular-offset model was used
    phase_opt_nulls = None
    phase_fit_table = None
    PHASE_FIT_MODEL = C["PHASE_FIT_MODEL"]

    # angular-error conversion settings (needs the radius axis)
    ang_p = float(C["ANG_EXPONENT"])
    ang_ref_mm = None
    if C["ANG_NORM"] and RADIUS_MODE:
        ang_ref_mm = float(C["ANG_REF_MM"]) if C["ANG_REF_MM"] else max(sp_r0, sp_r1)
    ang_c_used = None          # offset c that the fit used (fixed or fitted), reused for the angular-error curve
    ang_fit_curve = None       # k * E(r) of the fitted geometry, in the same units as the angular error
    ang_opt_curve = None       # k * E(r) of the optimal alignment, same units

    if C["PHASE_FIT_TRACKING_ERROR"]:
        if not RADIUS_MODE:
            say("Tracking-error fit needs an active record profile (radius axis). Fit not applied.", "warn")
        elif PHASE_FIT_MODEL == "geometry" and not C["PHASE_FIT_PIVOT_SPINDLE_MM"]:
            say("Geometry fit needs the pivot-to-spindle distance. Fit not applied.", "warn")
        elif PHASE_FIT_MODEL == "geometry_L" and not C["PHASE_FIT_EFF_LENGTH_MM"]:
            say("Fixed-length geometry fit needs the effective length. Fit not applied.", "warn")
        else:
            TRIM = C["PHASE_FIT_TRIM_S"]
            _fit_mask = (times >= TRIM) & (times <= times[-1] - TRIM) & np.isfinite(radius_mm)
            _rf = radius_mm[_fit_mask]
            _pf = phase_smooth[_fit_mask]
            if _rf.size < 10:
                say("Tracking-error fit: not enough in-band frames to fit. Fit not applied.", "warn")
            else:
                _r_lo, _r_hi = min(sp_r0, sp_r1), max(sp_r0, sp_r1)
                _c_fixed = C["PHASE_FIT_OFFSET_DEG"]
                _unit = "deg" if PHASE_UNIT == "degrees" else "rad"
                # Response weight: phase = c + k * W(r) * E(r) with W = (r_ref/r)^p, so k is the phase per degree
                # of angular error at r_ref. W = 1 (R15 behaviour) when the conversion is off or in shape mode.
                _ang_geo = (ang_ref_mm is not None) and PHASE_FIT_MODEL != "shape"
                # With the conversion on, the offset is a constant ANGULAR error delta (cartridge zenith / record
                # tilt), which gives a phase that varies as 1/r: phase = W(r) * (k*E(r) + delta). delta is in the
                # same units as k*E, i.e. degrees of phase at r_ref (zenith error in degrees = delta / k).
                # Without the conversion (R15) it stays a constant phase offset c.
                # Known arm offset angle: fix alpha to the offset-angle input and FIT the zenith offset instead
                # (they are exactly degenerate, so only one of the two can be fitted).
                _alpha_fixed = bool(C.get("PHASE_FIT_ALPHA_KNOWN")) and _ang_geo
                if _alpha_fixed:
                    _c_fixed = None
                _offname = "Zenith offset" if _ang_geo else "Offset"
                _offtxt = "zenith offset" if _ang_geo else "offset"

                def _W(r):
                    if not _ang_geo:
                        return 1.0
                    return (ang_ref_mm / np.maximum(np.asarray(r, float), 1.0)) ** ang_p
                # geometry: pivot-to-spindle D is fixed, effective length L = D + overhang is fitted.
                # geometry_L: effective length L is fixed (e.g. SME), D = L - overhang is fitted
                # (the arm base is adjustable). In both cases the free parameter is the overhang.
                _fixL = (PHASE_FIT_MODEL == "geometry_L")
                _D0 = float(C["PHASE_FIT_PIVOT_SPINDLE_MM"] or 0.0)
                _L0 = float(C["PHASE_FIT_EFF_LENGTH_MM"] or 0.0)

                def _LD(oh):
                    """(effective length, pivot-to-spindle) for a given overhang."""
                    return (_L0, _L0 - oh) if _fixL else (_D0 + oh, _D0)

                def _E_deg(r, L, D, alpha_deg):
                    alpha = np.radians(alpha_deg)
                    cosd = np.clip((L**2 + r**2 - D**2) / (2.0 * L * r), -1.0, 1.0)
                    return np.degrees(np.pi / 2.0 - alpha - np.arccos(cosd))

                if PHASE_FIT_MODEL == "shape":
                    if _c_fixed is None:
                        def _resid(params, r, y):
                            c, k, n1, n2 = params
                            return c + k * (r - n1) * (r - n2) / r - y
                        _p0 = [float(np.mean(_pf)), 0.01, C["PHASE_FIT_N1_GUESS_MM"], C["PHASE_FIT_N2_GUESS_MM"]]
                        _bounds = ([-np.inf, -np.inf, _r_lo, _r_lo], [np.inf, np.inf, _r_hi, _r_hi])
                    else:
                        def _resid(params, r, y):
                            k, n1, n2 = params
                            return _c_fixed + k * (r - n1) * (r - n2) / r - y
                        _p0 = [0.01, C["PHASE_FIT_N1_GUESS_MM"], C["PHASE_FIT_N2_GUESS_MM"]]
                        _bounds = ([-np.inf, _r_lo, _r_lo], [np.inf, _r_hi, _r_hi])
                    # keep start guesses inside bounds
                    _p0 = [float(np.clip(v, lo, hi)) for v, lo, hi in zip(_p0, _bounds[0], _bounds[1])]
                else:
                    _oh_lo, _oh_hi = -20.0, 80.0
                    _al_lo, _al_hi = 5.0, 35.0
                    if _alpha_fixed:
                        _a0 = float(C["PHASE_FIT_OFFSET_ANGLE_GUESS_DEG"])
                        _al_lo, _al_hi = _a0 - 1e-6, _a0 + 1e-6      # offset angle held at the given value
                    if _c_fixed is None:
                        def _resid(params, r, y):
                            c, k, oh, alpha = params
                            return _W(r) * (c + k * _E_deg(r, *_LD(oh), alpha)) - y
                        _p0 = [float(np.mean(_pf)), 0.03, C["PHASE_FIT_OVERHANG_GUESS_MM"], C["PHASE_FIT_OFFSET_ANGLE_GUESS_DEG"]]
                        _bounds = ([-np.inf, -np.inf, _oh_lo, _al_lo], [np.inf, np.inf, _oh_hi, _al_hi])
                    else:
                        def _resid(params, r, y):
                            k, oh, alpha = params
                            return _W(r) * (_c_fixed + k * _E_deg(r, *_LD(oh), alpha)) - y
                        _p0 = [0.03, C["PHASE_FIT_OVERHANG_GUESS_MM"], C["PHASE_FIT_OFFSET_ANGLE_GUESS_DEG"]]
                        _bounds = ([-np.inf, _oh_lo, _al_lo], [np.inf, _oh_hi, _al_hi])
                _starts = [_p0]
                if ang_ref_mm is not None and PHASE_FIT_MODEL != "shape":
                    # Weighted (angular-error) geometry fit: k is the phase per degree at r_ref, typically ~1, and
                    # R15's start k = 0.03 can end in the inverted local minimum (negative k, angle at its limit).
                    # Add a start from a coarse grid over (overhang, offset angle) with k (and c) solved in closed
                    # form, preferring positive k (curve with its ends up). R15's own start is kept as a second
                    # start and the lower-error result wins, so this is never worse than the R15 start.
                    def _grid_start():
                        _sub = max(1, len(_rf) // 400)
                        _rs = _rf[::_sub]
                        _ys = _pf[::_sub] / _W(_rs)      # angular-error domain: y/W = k*E + delta
                        _best = None
                        for _oh in np.linspace(_oh_lo, _oh_hi, 61):
                            _Lg, _Dg = _LD(_oh)
                            if _Dg <= 0 or _Lg <= 0:
                                continue
                            for _al in np.linspace(_al_lo, _al_hi, 61):
                                _B = _E_deg(_rs, _Lg, _Dg, _al)
                                if _c_fixed is None:
                                    _Bc = _B - _B.mean()
                                    _den = float(_Bc @ _Bc)
                                    if _den <= 0:
                                        continue
                                    _k = float(_Bc @ (_ys - _ys.mean())) / _den
                                    _c = float(_ys.mean() - _k * _B.mean())
                                else:
                                    _den = float(_B @ _B)
                                    if _den <= 0:
                                        continue
                                    _c = float(_c_fixed)
                                    _k = float(_B @ (_ys - _c)) / _den
                                _cand = (_k <= 0, float(np.sum((_c + _k * _B - _ys) ** 2)), _oh, _al, _k, _c)
                                if _best is None or _cand[:2] < _best[:2]:
                                    _best = _cand
                        _oh, _al, _k, _c = _best[2:]
                        return [_c, _k, _oh, _al] if _c_fixed is None else [_k, _oh, _al]
                    try:
                        _starts.append(_grid_start())
                    except Exception:
                        pass
                try:
                    _res = None
                    for _s0 in _starts:
                        # residual taken in the angular-error domain: (model - phase) / W(r) = k*E(r) - (phase - c)*(r/r_ref)^p
                        _try = opt.least_squares(lambda _pp, _r, _y: _resid(_pp, _r, _y) / _W(_r), _s0,
                                                 args=(_rf, _pf), bounds=_bounds)
                        if _res is None or _try.cost < _res.cost:
                            _res = _try
                    _ss_res = float(np.sum(_res.fun ** 2))
                    _ss_tot = float(np.sum((_pf - np.mean(_pf)) ** 2))
                    r2_fit = 1.0 - _ss_res / _ss_tot if _ss_tot > 0 else float('nan')

                    if PHASE_FIT_MODEL == "shape":
                        if _c_fixed is None:
                            c_fit, k_fit, n1_fit, n2_fit = _res.x
                        else:
                            k_fit, n1_fit, n2_fit = _res.x
                            c_fit = _c_fixed
                        if n1_fit > n2_fit:
                            n1_fit, n2_fit = n2_fit, n1_fit
                        phase_fit_curve = c_fit + k_fit * (radius_mm - n1_fit) * (radius_mm - n2_fit) / radius_mm
                        ang_c_used = float(c_fit)
                        phase_fit_active = True
                        phase_fit_nulls = (n1_fit, n2_fit)
                        phase_fit_note = (f"Tracking-error fit (shape): nulls at {n1_fit:.1f} & {n2_fit:.1f} mm, "
                                          f"offset {c_fit:+.3f} {_unit}"
                                          + (" (fixed)" if _c_fixed is not None else "") + f", R²={r2_fit:.2f}")
                        say(phase_fit_note)
                        phase_fit_table = [
                            ("Model", "shape", None),
                            ("Nulls (mm)", f"{n1_fit:.1f} & {n2_fit:.1f}", None),
                            (f"Offset ({_unit})", f"{c_fit:+.3f}" + (" (fixed)" if _c_fixed is not None else ""), None),
                            ("R²", f"{r2_fit:.2f}", None),
                        ]
                        if _c_fixed is None:
                            say("The curve shape alone only fixes k and n1×n2; the offset and (n1+n2) trade off "
                                "one-for-one, so the individual null positions are approximate. Set a known "
                                "constant phase offset to make n1, n2 identifiable.", "warn")
                    else:
                        if _c_fixed is None:
                            c_fit, k_fit, oh_fit, alpha_fit = _res.x
                        else:
                            k_fit, oh_fit, alpha_fit = _res.x
                            c_fit = _c_fixed
                        L_fit, D_fit = _LD(oh_fit)
                        phase_fit_curve = _W(radius_mm) * (c_fit + k_fit * _E_deg(radius_mm, L_fit, D_fit, alpha_fit))
                        ang_c_used = float(c_fit)
                        if _ang_geo:
                            ang_fit_curve = k_fit * _E_deg(radius_mm, L_fit, D_fit, alpha_fit) + c_fit   # c_fit = constant angular (zenith) offset
                            zen_used = (float(c_fit), float(k_fit), _c_fixed is not None)
                        phase_fit_active = True
                        if _fixL:
                            _mname = "geometry, fixed length"
                            _given_txt = (f"effective length {L_fit:.2f} mm (given), pivot-spindle {D_fit:.2f} mm "
                                          f"(fitted, overhang {oh_fit:+.2f} mm)")
                            _geo_rows = [("Effective length (mm)", f"{L_fit:.2f} (given)"),
                                         ("Pivot-spindle (mm)", f"{D_fit:.2f}"),
                                         ("Overhang (mm)", f"{oh_fit:+.2f}")]
                        else:
                            _mname = "geometry"
                            _given_txt = (f"pivot-spindle {D_fit:.1f} mm (given), overhang {oh_fit:+.2f} mm "
                                          f"(Leff {L_fit:.2f} mm)")
                            _geo_rows = [("Pivot-spindle (mm)", f"{D_fit:.1f} (given)"),
                                         ("Overhang (mm)", f"{oh_fit:+.2f}"),
                                         ("Leff (mm)", f"{L_fit:.2f}")]
                        _klabel = (f"k at r_ref ({_unit} phase per deg)" if _ang_geo else f"k ({_unit} phase per deg)")
                        _geo_rows.append((_klabel, f"{k_fit:.3f}"))
                        if _c_fixed is None and not _alpha_fixed:
                            # Offset angle is exactly degenerate with the free constant c (alpha only enters E(r)
                            # as an additive constant), so nulls are not determined by the curve shape.
                            phase_fit_note = f"Tracking-error fit ({_mname}): {_given_txt}, R²={r2_fit:.2f}"
                            say(phase_fit_note)
                            phase_fit_table = ([("Model", _mname, None)]
                                               + [(lbl, fv, None) for lbl, fv in _geo_rows]
                                               + [("Offset angle", "indeterminate", None),
                                                  ("Nulls (mm)", "indeterminate", None),
                                                  ("R²", f"{r2_fit:.2f}", None)])
                            say("With a free constant offset, offset angle and the constant offset are fully "
                                "degenerate: every offset angle can be matched by some offset to give an "
                                "identical curve. Null positions are therefore not shown. Set a known constant "
                                "phase offset to make offset angle and nulls identifiable.", "warn")
                        else:
                            _disc = D_fit**2 - (L_fit * np.cos(np.radians(alpha_fit)))**2
                            if _disc >= 0:
                                _s = np.sqrt(_disc)
                                n1_fit = L_fit * np.sin(np.radians(alpha_fit)) - _s
                                n2_fit = L_fit * np.sin(np.radians(alpha_fit)) + _s
                                phase_fit_nulls = (n1_fit, n2_fit)
                                _null_txt = f"nulls {n1_fit:.1f} & {n2_fit:.1f} mm, "
                            else:
                                _null_txt = "no real null radii for this fitted geometry, "
                            phase_fit_note = (f"Tracking-error fit ({_mname}): {_given_txt}, "
                                              f"offset angle {alpha_fit:.2f} deg{' (given)' if _alpha_fixed else ''}, " + _null_txt +
                                              f"{_offtxt} {c_fit:+.3f} {_unit} ({'fitted' if _alpha_fixed else 'fixed'}), R²={r2_fit:.2f}")
                            say(phase_fit_note)
                            _fitted_null_str = (f"{n1_fit:.1f} & {n2_fit:.1f}" if _disc >= 0 else "none (complex)")
                            phase_fit_table = ([(lbl, fv, None) for lbl, fv in _geo_rows]
                                               + [("Offset angle (deg)", f"{alpha_fit:.2f}" + (" (given)" if _alpha_fixed else ""), None),
                                                  ("Nulls (mm)", _fitted_null_str, None),
                                                  (f"{_offname} ({_unit})", f"{c_fit:+.3f} ({'fitted' if _alpha_fixed else 'fixed'})", None),
                                                  ("R²", f"{r2_fit:.2f}", None)])

                            # Optimal (Loefgren A) comparison curve for the chosen groove range, same k and c.
                            # Nulls n = L sin(a) +/- sqrt(D^2 - (L cos a)^2) must equal the targets t1, t2:
                            #   L sin(a) = (t1+t2)/2   and   D^2 - (L cos a)^2 = ((t2-t1)/2)^2.
                            # geometry: D fixed -> solve L and a.  geometry_L: L fixed -> solve a and D.
                            if C["PHASE_FIT_SHOW_OPTIMAL"]:
                                _lg_r1 = float(C["PHASE_FIT_LOFGREN_R1_MM"]) if C["PHASE_FIT_LOFGREN_R1_MM"] else min(sp_r0, sp_r1)
                                _lg_r2 = float(C["PHASE_FIT_LOFGREN_R2_MM"]) if C["PHASE_FIT_LOFGREN_R2_MM"] else max(sp_r0, sp_r1)
                                _lg_s = np.sqrt(0.5)
                                _t1 = 2.0 / ((1 + _lg_s) / _lg_r1 + (1 - _lg_s) / _lg_r2)
                                _t2 = 2.0 / ((1 - _lg_s) / _lg_r1 + (1 + _lg_s) / _lg_r2)
                                _half_sep = (_t2 - _t1) / 2.0
                                _Lsin = (_t1 + _t2) / 2.0
                                _opt_ok = False
                                if _fixL:
                                    if _Lsin <= _L0:
                                        L_opt = _L0
                                        alpha_opt = float(np.degrees(np.arcsin(_Lsin / _L0)))
                                        D_opt = float(np.sqrt(_half_sep**2 + (_L0 * np.cos(np.radians(alpha_opt)))**2))
                                        _opt_ok = True
                                else:
                                    _Lcos_sq = _D0**2 - _half_sep**2
                                    if _Lcos_sq >= 0:
                                        L_opt = float(np.hypot(_Lsin, np.sqrt(_Lcos_sq)))
                                        alpha_opt = float(np.degrees(np.arctan2(_Lsin, np.sqrt(_Lcos_sq))))
                                        D_opt = _D0
                                        _opt_ok = True
                                if _opt_ok:
                                    oh_opt = L_opt - D_opt
                                    phase_opt_curve = _W(radius_mm) * ((0.0 if _ang_geo else c_fit) + k_fit * _E_deg(radius_mm, L_opt, D_opt, alpha_opt))   # pure theory: no zenith offset
                                    if _ang_geo:
                                        ang_opt_curve = k_fit * _E_deg(radius_mm, L_opt, D_opt, alpha_opt)
                                        if C["PHASE_FIT_SHOW_ZENITH_CURVE"]:
                                            # theory shifted by the constant angular (zenith) offset of the fit
                                            ang_zen_curve = ang_opt_curve + c_fit
                                            phase_zen_curve = _W(radius_mm) * ang_zen_curve
                                    phase_opt_nulls = (_t1, _t2)
                                    _rng = (f"Löfgren A for {_lg_r1:.2f}-{_lg_r2:.2f} mm groove range "
                                            f"-> nulls {_t1:.1f} & {_t2:.1f} mm")
                                    if _fixL:
                                        say(f"Optimal alignment ({_rng}) at effective length {L_opt:.2f} mm: set "
                                            f"pivot-to-spindle to {D_opt:.2f} mm (overhang {oh_opt:+.2f} mm), "
                                            f"offset angle {alpha_opt:.2f} deg "
                                            f"(fitted: pivot-spindle {D_fit:.2f} mm, offset angle {alpha_fit:.2f} deg)")
                                    else:
                                        say(f"Optimal alignment ({_rng}) at pivot-spindle {D_opt:.1f} mm: "
                                            f"overhang {oh_opt:+.2f} mm (Leff {L_opt:.2f} mm), "
                                            f"offset angle {alpha_opt:.2f} deg "
                                            f"(fitted: overhang {oh_fit:+.2f} mm, offset angle {alpha_fit:.2f} deg)")
                                    _col = {"Overhang (mm)": f"{oh_opt:+.2f}",
                                            "Offset angle (deg)": f"{alpha_opt:.2f}",
                                            "Nulls (mm)": f"{_t1:.1f} & {_t2:.1f} (target)",
                                            f"{_offname} ({_unit})": ("0 (theory)" if _ang_geo else "same (fixed)"),
                                            "R²": "—"}
                                    if _fixL:
                                        _col["Effective length (mm)"] = f"{L_opt:.2f} (given)"
                                        _col["Pivot-spindle (mm)"] = f"{D_opt:.2f}"
                                    else:
                                        _col["Pivot-spindle (mm)"] = f"{D_opt:.1f} (given)"
                                        _col["Leff (mm)"] = f"{L_opt:.2f}"
                                    _col[_klabel] = "same"
                                    phase_fit_table = [(lbl, fv, _col.get(lbl)) for lbl, fv, _ in phase_fit_table]
                                elif _fixL:
                                    say(f"Optimal alignment: target nulls {_t1:.1f}/{_t2:.1f} mm cannot be reached "
                                        f"with effective length {_L0:.2f} mm. No solution.", "warn")
                                else:
                                    say(f"Optimal alignment: target nulls {_t1:.1f}/{_t2:.1f} mm are farther "
                                        f"apart than pivot-spindle distance {_D0:.1f} mm allows. No solution.", "warn")
                except Exception as _e:
                    say(f"Tracking-error fit failed ({_e}). Fit not applied.", "warn")

    # ---------------- angular error: phase converted with the 1/r response ----------------
    # angular error (phase-equivalent at r_ref) = (phase - c) * (r / r_ref)^p
    ang_active = False
    ang_err = np.full(len(times), np.nan)
    ang_note = ""
    if C["ANG_NORM"]:
        if not RADIUS_MODE:
            say("Angular-error conversion needs an active record profile (radius axis). Not applied.", "warn")
        else:
            if ang_c_used is not None:
                _c_ang, _c_src = ang_c_used, ("fixed" if (zen_used[2] if zen_used else C["PHASE_FIT_OFFSET_DEG"] is not None) else "fitted")
            else:
                _c_ang = float(C["PHASE_FIT_OFFSET_DEG"]) if C["PHASE_FIT_OFFSET_DEG"] is not None else 0.0
                _c_src = "set manually" if C["PHASE_FIT_OFFSET_DEG"] is not None else "assumed 0, no fit"
            _wr = (np.maximum(radius_mm, 1.0) / ang_ref_mm) ** ang_p
            if PHASE_FIT_MODEL != "shape":
                ang_err = phase_smooth * _wr                   # includes the constant zenith offset
                _ang_form = "phase x (r / r_ref)^p"
            else:
                ang_err = (phase_smooth - _c_ang) * _wr
                _ang_form = "(phase - c) x (r / r_ref)^p"
            ang_active = True
            _u = "deg" if PHASE_UNIT == "degrees" else "rad"
            ang_note = (f"Angular error: {_ang_form.replace('r_ref', f'{ang_ref_mm:.2f} mm').replace('^p', f'^{ang_p:g}')}, offset = {_c_ang:+.3f} {_u} ({_c_src}); "
                        f"units: {_u} of phase this angular error gives at {ang_ref_mm:.2f} mm")
            say(ang_note)
            if ang_p != 0 and PHASE_FIT_MODEL == "shape":
                say("The shape fit does not use the 1/r response, so no model curve is drawn on the angular-error plot.", "warn")

    L_smooth = smooth_signal(level_L, sr_frames, SMOOTH_MS)
    R_smooth = smooth_signal(level_R, sr_frames, SMOOTH_MS)
    harmonic_db_L_smooth = smooth_signal(np.nan_to_num(harmonic_db_L, nan=MIN_RATIO_DB), sr_frames, SMOOTH_MS)
    harmonic_db_R_smooth = smooth_signal(np.nan_to_num(harmonic_db_R, nan=MIN_RATIO_DB), sr_frames, SMOOTH_MS)

    # ---------------- radius-normalised H2 ----------------
    harm_norm_active = False
    norm_ref_mm = None
    harmonic_pct_norm_L = np.full(len(times), np.nan)
    harmonic_pct_norm_R = np.full(len(times), np.nan)
    harmonic_db_norm_L_smooth = np.full(len(times), np.nan)
    harmonic_db_norm_R_smooth = np.full(len(times), np.nan)
    harm_norm_note = ""

    def _pct_to_db(pct, valid):
        db = 20.0 * np.log10(np.maximum(np.nan_to_num(pct, nan=0.0) / 100.0, EPS))
        db[~valid] = MIN_RATIO_DB
        return db

    if C["HARM_RADIUS_NORM"] and MODE != "broadband":
        if not RADIUS_MODE:
            say("Radius-normalised H2 needs an active record profile (radius axis). Not applied.", "warn")
        else:
            norm_ref_mm = float(C["HARM_NORM_REF_MM"]) if C["HARM_NORM_REF_MM"] else max(sp_r0, sp_r1)
            r_clip = np.clip(radius_mm, min(sp_r0, sp_r1), max(sp_r0, sp_r1))
            norm_factor = r_clip / norm_ref_mm
            harmonic_pct_norm_L = harmonic_linear_L * 100.0 * norm_factor
            harmonic_pct_norm_R = harmonic_linear_R * 100.0 * norm_factor
            harmonic_db_norm_L_smooth = smooth_signal(_pct_to_db(harmonic_pct_norm_L, harmonic_valid_L), sr_frames, SMOOTH_MS)
            harmonic_db_norm_R_smooth = smooth_signal(_pct_to_db(harmonic_pct_norm_R, harmonic_valid_R), sr_frames, SMOOTH_MS)
            harm_norm_active = True
            harm_norm_note = f"H2 normalised: % x r / {norm_ref_mm:.2f} mm"
            say(harm_norm_note)

    # ---------------- time delay ----------------
    delay_sec = np.full_like(phase_vals, np.nan, dtype=float)
    valid_freq = (~np.isnan(tracked_freqs)) & (tracked_freqs > 0)
    delay_sec[valid_freq] = phase_vals[valid_freq] / (2 * np.pi * tracked_freqs[valid_freq])
    delay_us_smooth = smooth_signal(delay_sec, sr_frames, SMOOTH_MS) * 1e6

    # ---------------- CSV ----------------
    n = len(times)
    df = pd.DataFrame({
        "time_s": times,
        "radius_mm": radius_mm,
        "phase_deg": phase_smooth,
        "coherence": coherence_vals,
        "level_L_dbfs": L_smooth,
        "level_R_dbfs": R_smooth,
        "tracked_freq_hz": tracked_freqs,
        "delay_us": delay_us_smooth,
        "harmonic_linear_L": harmonic_linear_L,
        "harmonic_db_L": harmonic_db_L_smooth,
        "harmonic_valid_L": harmonic_valid_L.astype(int),
        "harmonic_linear_R": harmonic_linear_R,
        "harmonic_db_R": harmonic_db_R_smooth,
        "harmonic_valid_R": harmonic_valid_R.astype(int),
        "phase_fit": phase_fit_curve if phase_fit_active else np.full(n, np.nan),
        "angular_error_phase_equiv": ang_err,
        "angular_error_fit": ang_fit_curve if ang_fit_curve is not None else np.full(n, np.nan),
        "harmonic_pct_norm_L": harmonic_pct_norm_L,
        "harmonic_db_norm_L": harmonic_db_norm_L_smooth,
        "harmonic_pct_norm_R": harmonic_pct_norm_R,
        "harmonic_db_norm_R": harmonic_db_norm_R_smooth,
    })
    csv_bytes = df.to_csv(index=False, float_format="%.6f").encode("utf-8")
    del df

    return dict(
        log=log, MODE=MODE, PHASE_UNIT=PHASE_UNIT, SMOOTH_MS=SMOOTH_MS, display_name=display_name,
        RADIUS_MODE=RADIUS_MODE, prof_label=prof_label, sp_side=sp_side, prof_markers=prof_markers,
        sp_r0=sp_r0, sp_r1=sp_r1, duration_s=duration_s, times=times, x_plot=x_plot,
        phase_smooth=phase_smooth, phase_fit_active=phase_fit_active, phase_fit_curve=phase_fit_curve,
        phase_fit_nulls=phase_fit_nulls, phase_opt_curve=phase_opt_curve, phase_zen_curve=phase_zen_curve, zen_used=zen_used, ang_ref_for_note=ang_ref_mm, ang_zen_curve=ang_zen_curve, phase_opt_nulls=phase_opt_nulls,
        phase_fit_table=phase_fit_table, phase_fit_note=phase_fit_note, radius_note=radius_note,
        harm_norm_note=harm_norm_note, harm_norm_active=harm_norm_active, norm_ref_mm=norm_ref_mm,
        harmonic_db_norm_L_smooth=harmonic_db_norm_L_smooth, harmonic_db_norm_R_smooth=harmonic_db_norm_R_smooth,
        harmonic_db_L_smooth=harmonic_db_L_smooth, harmonic_db_R_smooth=harmonic_db_R_smooth,
        harmonic_db_L=harmonic_db_L, harmonic_db_R=harmonic_db_R,
        harmonic_valid_L=harmonic_valid_L, harmonic_valid_R=harmonic_valid_R,
        delay_us_smooth=delay_us_smooth, L_smooth=L_smooth, R_smooth=R_smooth,
        tracked_freqs=tracked_freqs, csv_bytes=csv_bytes,
        ang_active=ang_active, ang_err=ang_err, ang_fit_curve=ang_fit_curve, ang_opt_curve=ang_opt_curve,
        ang_ref_mm=ang_ref_mm, ang_p=ang_p, ang_note=ang_note,
    )


# ---------------------------
# Plotting (R15 layout, dark theme)
# ---------------------------
def _style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": "#888888", "axes.labelcolor": FG, "text.color": FG,
        "xtick.color": FG, "ytick.color": FG, "grid.color": "#4a4a4a",
        "legend.facecolor": "#2a2a2a", "legend.edgecolor": "#555555", "legend.labelcolor": FG,
        "font.size": BASE_FONT - 2, "axes.titlesize": BASE_FONT, "axes.labelsize": BASE_FONT - 1,
        "legend.fontsize": BASE_FONT - 3,
    })


def _grid(ax):
    ax.grid(which='major', linestyle='-', linewidth=0.8)
    ax.grid(which='minor', linestyle=':', linewidth=0.5)
    ax.minorticks_on()


def configure_x_axis(ax, R, C):
    if R["RADIUS_MODE"]:
        lo, hi = sorted(C["RADIUS_XLIM_MM"]) if C["RADIUS_XLIM_MM"] else sorted((R["sp_r0"], R["sp_r1"]))
        ax.set_xlim((hi, lo) if C["RADIUS_INVERT_X"] else (lo, hi))
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.set_xlabel("Groove radius (mm)")
    else:
        ax.set_xlabel("Time (s)")
        t = R["times"]
        if len(t) > 1:
            step = (t[-1] - t[0]) / 20.0
            ax.xaxis.set_major_locator(MultipleLocator(step))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))


def draw_phase(ax, legend_y, R, C):
    x = R["x_plot"]
    handles = [ax.plot(x, R["phase_smooth"], label=f"Phase ({R['MODE']}) smoothed {R['SMOOTH_MS']:.0f} ms",
                       linewidth=1.5, color="C0")[0]]
    if R["phase_fit_active"]:
        handles.append(ax.plot(x, R["phase_fit_curve"], '--', color=FIT_COLOR, linewidth=1.2,
                               label="Theoretical curve (fitted model)")[0])
        if R["phase_fit_nulls"] is not None:
            n1, n2 = R["phase_fit_nulls"]
            for nv in (n1, n2):
                ax.axvline(x=nv, color=FIT_COLOR, linewidth=1.0, linestyle=':')
            handles.append(ax.plot([], [], color=FIT_COLOR, linewidth=1.0, linestyle=':',
                                   label=f"Fitted nulls ({n1:.1f}, {n2:.1f} mm)")[0])
    if R["phase_opt_curve"] is not None:
        t1, t2 = R["phase_opt_nulls"]
        handles.append(ax.plot(x, R["phase_opt_curve"], color='#5ad25a', linewidth=1.3, linestyle=':',
                               label=f"Optimal alignment curve (nulls {t1:.1f}, {t2:.1f} mm)")[0])
    _zen_note(ax, R)
    if R.get("phase_zen_curve") is not None:
        handles.append(ax.plot(x, R["phase_zen_curve"], color='#e0a030', linewidth=1.3, linestyle='-.',
                               label="Optimal curve + constant zenith angular error")[0])
    ax.set_ylabel(f"Phase difference ({R['PHASE_UNIT']})")
    _grid(ax)
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, legend_y),
              ncol=min(len(handles), 2), frameon=True)
    if C["Y_AXIS_LIMITS"]:
        ax.set_ylim(C["Y_AXIS_LIMITS"])


def _zen_note(ax, R):
    """Note in the axes: the constant zenith (angular) offset used by the fit."""
    z = R.get("zen_used")
    if not z:
        return
    d, k, fixed = z
    ref = R.get("ang_ref_for_note")
    txt = (f"Zenith offset used: {d:+.3f} deg of phase at {ref:.0f} mm ({'set manually' if fixed else 'fitted'})"
           + (f" = {d / k:+.3f} deg of angle" if abs(k) > 1e-9 else ""))
    ax.text(0.01, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#7fc8ff",
            bbox=dict(boxstyle="round,pad=0.3", fc="#00000080", ec="#7fc8ff", lw=0.6))


def draw_ang(ax, legend_y, R, C):
    """Angular error = (phase - c) x (r/r_ref)^p, in degrees of phase at r_ref, with the fitted and optimal
    geometry curves (k x E(r)) in the same units."""
    x = R["x_plot"]
    ref, p = R["ang_ref_mm"], R["ang_p"]
    handles = [ax.plot(x, R["ang_err"], linewidth=1.5, color="C0",
                       label=(f"Angular error: phase x (r/{ref:.0f} mm)^{p:g}" if R.get("ang_fit_curve") is not None
                              else f"Angular error: (phase - c) x (r/{ref:.0f} mm)^{p:g}"))[0]]
    if R["ang_fit_curve"] is not None:
        handles.append(ax.plot(x, R["ang_fit_curve"], '--', color=FIT_COLOR, linewidth=1.2,
                               label="Geometry curve (fitted)")[0])
        if R["phase_fit_nulls"] is not None:
            for nv in R["phase_fit_nulls"]:
                ax.axvline(x=nv, color=FIT_COLOR, linewidth=1.0, linestyle=':')
    if R["ang_opt_curve"] is not None:
        t1, t2 = R["phase_opt_nulls"]
        handles.append(ax.plot(x, R["ang_opt_curve"], color='#5ad25a', linewidth=1.3, linestyle=':',
                               label=f"Optimal alignment (nulls {t1:.1f}, {t2:.1f} mm)")[0])
    if R.get("ang_zen_curve") is not None:
        handles.append(ax.plot(x, R["ang_zen_curve"], color='#e0a030', linewidth=1.3, linestyle='-.',
                               label="Optimal curve + constant zenith angular error")[0])
    _zen_note(ax, R)
    ax.axhline(0.0, color="#888888", linewidth=0.8)
    unit = "deg" if R["PHASE_UNIT"] == "degrees" else "rad"
    ax.set_ylabel(f"Angular error ({unit}, phase-equiv. at {ref:.0f} mm)")
    _grid(ax)
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, legend_y),
              ncol=min(len(handles), 2), frameon=True)


def draw_harm(ax, legend_y, R, C):
    x = R["x_plot"]
    if R["harm_norm_active"]:
        ref = R["norm_ref_mm"]
        ax.plot(x, R["harmonic_db_norm_L_smooth"], label=f"2nd harm L, norm. to {ref:.2f} mm", color='C3')
        ax.plot(x, R["harmonic_db_norm_R_smooth"], label=f"2nd harm R, norm. to {ref:.2f} mm", color='C4')
        if C["HARM_NORM_SHOW_RAW"]:
            ax.plot(x, R["harmonic_db_L_smooth"], label="2nd harm L, raw", color='C3', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.plot(x, R["harmonic_db_R_smooth"], label="2nd harm R, raw", color='C4', linestyle='--', linewidth=0.8, alpha=0.6)
    else:
        ax.plot(x, R["harmonic_db_L_smooth"], label="2nd harmonic L (dB rel. fund)", color='C3')
        ax.plot(x, R["harmonic_db_R_smooth"], label="2nd harmonic R (dB rel. fund)", color='C4')
    inv_L = ~R["harmonic_valid_L"]
    inv_R = ~R["harmonic_valid_R"]
    if R["MODE"] != "broadband":
        if inv_L.any():
            ax.plot(x[inv_L], R["harmonic_db_L"][inv_L], 'x', color='#999999', markersize=4, label='L invalid')
        if inv_R.any():
            ax.plot(x[inv_R], R["harmonic_db_R"][inv_R], '+', color='#cccccc', markersize=4, label='R invalid')
    ax.set_ylabel(f"2nd harm. (dB, norm. {R['norm_ref_mm']:.0f} mm)" if R["harm_norm_active"] else "2nd harmonic (dB)")
    _grid(ax)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_y), ncol=2, frameon=True)


def draw_markers(axes, R):
    if R["RADIUS_MODE"]:
        marks = [(r, f'{r} mm') for r in R["prof_markers"]]
    elif R["duration_s"] > 1000:
        marks = [(377, '377 s'), (1209, '1209 s')]
    elif R["duration_s"] > 800:
        marks = [(273, '273 s'), (872, '872 s')]
    else:
        marks = []
    for ax in axes:
        for xv, lab in marks:
            ax.axvline(x=xv, color='#ff5555', linewidth=1.2, linestyle='--', label=lab)


def caption_text(R):
    cap = f"File: {R['display_name']}    Script: {SCRIPT_NAME} {SCRIPT_REV}"
    if R["radius_note"]:
        cap += "\n" + R["radius_note"]
    if R["harm_norm_note"]:
        cap += "    " + R["harm_norm_note"]
    if R["phase_fit_note"]:
        cap += "\n" + R["phase_fit_note"]
    if R["ang_active"]:
        cap += "\n" + R["ang_note"]
    return cap


def _title(base, R, C):
    if R["RADIUS_MODE"]:
        base += f" vs radius, {R['prof_label']} side {R['sp_side']}"
    t = f"{base} — {C['TITLE_TEXT']}" if C["TITLE_TEXT"] else base
    return f"{t}\n{R['display_name']}"      # second line: the analysed file name


def fig_page1(R, C):
    _style()
    fig = plt.figure(constrained_layout=True, figsize=(12, 11))
    gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=[1, 1, 1, 0.7, 0.12])
    ax_phase = fig.add_subplot(gs[0, 0])
    ax_delay = fig.add_subplot(gs[1, 0], sharex=ax_phase)
    ax_level = fig.add_subplot(gs[2, 0], sharex=ax_phase)
    ax_harm = fig.add_subplot(gs[3, 0], sharex=ax_phase)
    ax_cap = fig.add_subplot(gs[4, 0])

    draw_phase(ax_phase, -0.22, R, C)
    ax_phase.set_title(_title(f"R-L phase. Positive R leads L, level, delay and 2nd-harmonic ({R['MODE']})", R, C))

    x = R["x_plot"]
    ln = ax_delay.plot(x, R["delay_us_smooth"], label="Time delay (µs)", color='C2')[0]
    ax_delay.set_ylabel("Delay (µs)")
    _grid(ax_delay)
    ax_delay.legend(handles=[ln], loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=True)

    lL = ax_level.plot(x, R["L_smooth"], label="Left dBFS", color='C0')[0]
    lR = ax_level.plot(x, R["R_smooth"], label="Right dBFS", color='C1', alpha=0.9)[0]
    ax_level.set_ylabel("Level (dBFS)")
    _grid(ax_level)
    ax_level.legend(handles=[lL, lR], loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True)

    draw_harm(ax_harm, -0.45 if R["harm_norm_active"] else -0.32, R, C)
    configure_x_axis(ax_harm, R, C)

    ax_cap.axis('off')
    ax_cap.text(0.5, 0.5, caption_text(R), ha='center', va='center', fontsize=8, color='#9a9a9a')
    draw_markers([ax_phase, ax_delay, ax_level, ax_harm], R)
    return fig


def fig_page2(R, C):
    _style()
    tbl = R["phase_fit_table"]
    table_h = 0.06 * len(tbl) + 0.06 if tbl else 0.0
    ang = R["ang_active"]
    if ang:
        fig = plt.figure(constrained_layout=True, figsize=(12, 13 + table_h * 9))
        gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=[1, 1, 1, max(table_h, 0.001), 0.09])
        ax_p = fig.add_subplot(gs[0, 0])
        ax_a = fig.add_subplot(gs[1, 0], sharex=ax_p)
        ax_h = fig.add_subplot(gs[2, 0], sharex=ax_p)
        ax_t = fig.add_subplot(gs[3, 0])
        ax_c = fig.add_subplot(gs[4, 0])
    else:
        fig = plt.figure(constrained_layout=True, figsize=(12, 9 + table_h * 9))
        gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[1, 1, max(table_h, 0.001), 0.09])
        ax_p = fig.add_subplot(gs[0, 0])
        ax_h = fig.add_subplot(gs[1, 0], sharex=ax_p)
        ax_t = fig.add_subplot(gs[2, 0])
        ax_c = fig.add_subplot(gs[3, 0])

    draw_phase(ax_p, -0.24, R, C)
    ax_p.set_title(_title(f"R-L phase difference (positive: R leads L), angular error and 2nd harmonic ({R['MODE']})"
                          if ang else f"R-L phase difference (positive: R leads L) and 2nd harmonic ({R['MODE']})", R, C))
    draw_harm(ax_h, -0.16 if R["harm_norm_active"] else -0.13, R, C)
    configure_x_axis(ax_h, R, C)
    ax_p.set_xlabel(ax_h.get_xlabel())
    if ang:
        draw_ang(ax_a, -0.24, R, C)
        ax_a.set_xlabel(ax_h.get_xlabel())
        draw_markers([ax_p, ax_a, ax_h], R)
    else:
        draw_markers([ax_p, ax_h], R)

    ax_t.axis('off')
    if tbl:
        has_opt = any(o is not None for _, _, o in tbl)
        col_labels = ["Fitted"] + (["Optimal (target)"] if has_opt else [])
        cells = [[fv] + ([ov if ov is not None else "—"] if has_opt else []) for _, fv, ov in tbl]
        rows = [lbl for lbl, _, _ in tbl]
        t = ax_t.table(cellText=cells, rowLabels=rows, colLabels=col_labels,
                       cellLoc='center', rowLoc='right', loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        t.scale(1, 1.4)
        t.auto_set_column_width(col=list(range(len(col_labels))))
        for cell in t.get_celld().values():
            cell.set_facecolor("#2a2a2a")
            cell.set_edgecolor("#555555")
            cell.get_text().set_color(FG)

    ax_c.axis('off')
    ax_c.text(0.5, 0.5, caption_text(R), ha='center', va='center', fontsize=8, color='#9a9a9a')
    return fig


def fig_freq(R, C):
    _style()
    fig = plt.figure(constrained_layout=True, figsize=(12, 2.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(R["x_plot"], R["tracked_freqs"], label="Tracked frequency", color='C2')
    configure_x_axis(ax, R, C)
    ax.set_ylabel("Frequency (Hz)")
    t = "Tracked dominant frequency per frame"
    ax.set_title((f"{t} — {C['TITLE_TEXT']}" if C["TITLE_TEXT"] else t) + f"\n{R['display_name']}")
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=1, frameon=True)
    return fig


def fig_to_png(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=130)
    plt.close(fig)
    return b.getvalue()


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Sperling Phase Analyser", layout="wide")
st.title("EXPERIMENTAL, may fail!! R-L phase, delay and 2nd harmonic vs groove radius")
st.caption("Upload a stereo needle-drop of a single-tone test record (Sperling, Clearaudio, or any record in time mode). "
           "The app tracks the R-L phase difference, delay, level and 2nd harmonic, maps time to groove radius, "
           "and fits a tracking-error curve to check your tonearm alignment.")

D = DEFAULTS


def opt_float(label, default, key, help=None):
    """Number input with an 'auto' checkbox meaning None."""
    use = st.checkbox(f"{label}: set manually", value=default is not None, key=key + "_on", help=help)
    # Always render the number box: inside a form, a checkbox cannot show/hide widgets before Run.
    val = st.number_input(label, value=float(default if default is not None else 0.0), key=key, format="%.3f")
    return val if use else None


GUIDE_MD = """
## What this app does

You upload a stereo recording (needle drop) of a **single-tone test record**. The app follows the tone through
the whole recording and shows, frame by frame:

- the **R-L phase difference** (positive = right channel leads left),
- the equivalent **time delay** between the channels (phase / (2 pi f)),
- the **level** of each channel (dBFS),
- the **2nd harmonic** of each channel relative to the tone (dB).

Time is converted to **groove radius** using the record profile (the tone band has constant pitch, so radius is
linear in time). A **tracking-error curve** is then fitted to the phase-versus-radius curve, so you can see where
your tonearm geometry puts its alignment nulls and compare with an optimal (Loefgren A) alignment.

## Quick start

1. Upload a stereo **WAV or FLAC** (FLAC is about half the size and uploads faster).
2. Pick the **Record** (Sperling or clearaudio). The **Tone frequency** box fills in the record's tone
   (Sperling 1000 Hz, clearaudio 3150 Hz).
3. Choose the **Tracking-error fit model** (top of the sidebar) that matches your arm, then check its fixed
   value in *Tracking-error fit*: the measured **Pivot-to-spindle (mm)** for a fixed-pivot arm, or the
   **Effective length (mm)** for an arm with a fixed effective length and an adjustable base. Only the fields
   of the chosen model are shown.
4. Press **Run analysis** (top of the sidebar). Long recordings take a while; a progress bar is shown.
5. Read the result tabs, the fit table and the processing log. Download the PNG pages or the CSV if you want them.

## The recording

- Must be **stereo** (two channels). Mono files are rejected.
- Record the whole tone band, and start recording before the needle drop so the start of the tone is clean.
- The app detects where the tone starts and ends from the signal level. Needle-drop clicks shorter than the
  *hold time* are ignored.
- Streamlit limits uploads to 200 MB by default. If a file is refused, use FLAC, or raise `server.maxUploadSize`
  in the Streamlit configuration.

## Settings

**Record and tone frequency (top of the sidebar, update immediately)**

- *Record*: Sperling, clearaudio, or "none (time axis)" for any other record (no radius, no fit).
- *Tone frequency*: the frequency that peak mode tracks. It follows the selected record and can be edited.
  A wrong value here makes the app track noise, which shows up as huge random phase values.

**Analysis**

- *Mode*: **peak** tracks the tone near the tone frequency (use this for single-tone records). **sweep** follows the
  strongest bin in a band (for sweeps). **broadband** takes a magnitude-weighted phase over a band (no 2nd harmonic).
- *Search window*: how far from the tone frequency the peak is searched, in Hz. Peak mode only.
- *Sweep band / Broadband band*: frequency limits for those two modes. The broadband default is 900-1100 Hz,
  so change it if you use broadband mode on the 3150 Hz record.
- *FFT size and Hop size*: analysis frame length and step. A larger FFT gives finer frequency resolution but
  coarser time resolution. Defaults 32768 and 2048.
- *Smoothing (ms)*: moving average over the curves. Default 18000 ms. Longer smoothing gives a calmer curve and
  makes the fit less sensitive to local wiggles, but hides fast changes. 0 turns it off.
- *Phase unit*, *Unwrap phase*, *Title text*, and optional *fixed phase y-axis limits*.

**Record profile**

- *Record side (rpm)*: **auto** picks the side whose band length matches the detected tone length.
- *Radius alignment*: **both** detects tone start and end and stretches the radius scale to fit (this absorbs
  turntable speed error, shown as "speed" in the log). **onset** detects the start and assumes the nominal
  duration. **manual** starts at *Manual tone start (s)*.
- *X-axis min / max*, and *Outer radius on the left* (playback order).
- *Allowed span deviation*: if the detected tone length differs from every side of the profile by more than this,
  the app assumes the wrong profile or a partial recording and plots against **time** instead, with a warning.
- *Tone detect*: dB below the 90th-percentile level, and hold time. Lower these if the tone is not found.

**2nd harmonic**

- *Normalise to radius*: scales the harmonic percentage by r / r_ref, so values at different radii are comparable.
  The reference radius defaults to the outer radius of the band. Tick *set manually* to use another one.
- *Also show un-normalised*: draws the raw curves dashed for comparison.

**Tracking-error fit**

- *Tracking-error fit model* (top of the sidebar, next to Record): choose the one that matches how your arm
  is built. It is either-or: the fields of the other models are hidden. Both geometry models are the exact
  pivoted-arm model and derive the null radii from the fitted geometry.
  - **Geometry: fixed pivot-to-spindle** is for an arm whose pivot-to-spindle distance is fixed. It fits the
    overhang (and so the effective length) and the headshell offset angle.
  - **Geometry: fixed effective length** is for an arm with a fixed effective length and an adjustable base,
    such as an SME. It fits the pivot-to-spindle distance (overhang = effective length minus pivot-to-spindle)
    and the offset angle. The optimal alignment then tells you which pivot-to-spindle distance to set.
  - **Shape** is an abstract fit with free null radii.
- *Pivot-to-spindle (mm)*: the fixed, measured value for the fixed-pivot model. Not used by the other models.
- *Effective length (mm)*: the arm's pivot-to-stylus distance from its specification, for the fixed-length
  model. Not used by the other models.
- *Start guesses*: overhang and offset angle (both geometry models), or the two nulls (shape). Only the
  starting point of the fit. In the fixed-length model the start for the pivot-to-spindle distance is the
  effective length minus the overhang guess.
- *Known constant phase offset c*: tick *set manually* and enter the value. Default 0. See the note below.
- *Trim each end*: seconds removed from both ends of the file before fitting, to avoid edge effects. Default 10 s.
  The smoothing average is pulled toward zero in the first and last half-window of the file, so **keep the trim
  at least half the smoothing value** (default 18 s smoothing needs 9 s or more; 40 s smoothing needs 20 s or
  more). This assumes the file starts and ends on the tone. If the file has silence or noise before or after
  the tone, that noise is smeared into the edges of the tone band, so increase the trim by the length of that
  lead-in and run-out as well.
- *Show optimal (Loefgren A) curve*: plots the curve you would get with an optimal alignment and the same c.
  For the fixed-pivot model that is at the same pivot-to-spindle distance. For the fixed-length model it is at
  the same effective length, and the table gives the pivot-to-spindle distance that achieves it.
  *Loefgren R1 / R2* set the inner and outer groove radii it is
  optimised for (defaults 60.3 and 146.05 mm). Untick *set manually* to use the record's own band.

**Angular error (phase to angle conversion)**

- Phase is not the same thing as angular error. At a constant angular error the stylus reads the two groove
  walls at slightly different positions along the groove, so the channel phase grows as the groove gets
  slower and the recorded wavelength shorter: the same angular error gives about twice the phase at 60 mm as at
  120 mm. The app therefore converts phase to angular error as **(phase - c) x (r / r_ref)^p**.
- *Reference radius*: unticked = the outer radius of the band. The result is in **degrees of phase this angular
  error would produce at the outer radius**, so the outer end of the curve is unchanged and the inner part is
  scaled down. It is a relative unit: to get real degrees you also need the phase per degree of angular error at
  r_ref, which the fit table reports as *k at r_ref* (it is only as reliable as the model).
- *Exponent p*: 1 means a response proportional to 1/r (groove speed). 0 switches the conversion off. Values
  between 1 and 2 are worth trying if the converted curves for different cartridge settings still differ by a tilt.
- The geometry fits use the same response, so phase = c + k x (r_ref / r)^p x E(r), where E(r) is the exact
  tracking angle error of the arm. With the conversion on, the constant c and the offset angle are no longer
  exchangeable, because c is multiplied by r in the angular-error curve while the offset angle is not.
- The shape fit does not use the response; for that model the angular-error plot shows the converted data only.
- Tick off *Convert phase to angular error* to get the R15 fit back exactly (the fit table only gains a *k* row).\n- With the conversion on, the geometry fit also tries a start from a coarse grid over overhang and offset angle,\n  because *k* is now about 1 and the R15 start can end in an inverted solution (negative *k*, offset angle at its limit).

**Show page 2** adds a second figure with phase, angular error (with the fitted and optimal geometry curves),
2nd harmonic and the fit table.

### About the constant offset (zenith error)\n\nWith the angular-error conversion on, the offset is a constant *angular* error (cartridge zenith, record tilt): it\nproduces a phase that varies as 1/r, and the angular error is phase × (r/r_ref)^p minus this offset, in degrees of\nphase at r_ref (zenith in degrees = offset / k). It is exactly degenerate with the offset angle of the arm, so leaving\nit free gives an *effective* offset angle that includes any zenith error. With the conversion off it is the old\nconstant phase offset c.\n

The curve shape alone cannot separate a constant phase offset from the position of the nulls: different
combinations give an identical curve. So:

- With **c fixed** (ticked), overhang, offset angle and null positions are all determined, and the optimal
  curve is available.
- With **c free** (unticked), the app reports the fitted geometry (overhang and Leff, or the pivot-to-spindle
  distance for the fixed-length model), but offset angle and nulls are shown as
  *indeterminate*, and no optimal curve is drawn.

Only fix c to a value you know independently. The cause of a constant R-L phase offset (cartridge, cabling,
electronics, mastering) is not established by this analysis.

## Reading the results

- **Page 1** shows phase, delay, level and 2nd harmonic against groove radius (or time).
- **Page 2** shows phase with the fitted curve, the nulls, and the optimal curve, the angular error with the same
  curves in angular units, plus 2nd harmonic and the fit table.
- **Tracked frequency** shows the frequency found in each frame. It should be a flat line at the tone
  frequency. If it is not, the tone frequency or the record choice is wrong.
- **Fit table**: *Fitted* is what the recording shows. *Optimal (target)* is the Loefgren A alignment for the
  chosen groove range. R-squared tells you how well the model matches the smoothed phase curve.
- **Red dashed lines** mark the profile's marker radii (for Sperling, the IEC nulls at 120.9 and 66.0 mm).
- **Processing log** lists what was detected: tone start and end, side, speed deviation, fit result.
  Warnings are also shown at the top of the results.
- **Download time series CSV** contains time, radius, phase, coherence, levels, tracked frequency, delay, the
  harmonic columns, the fitted phase, the angular error (*angular_error_phase_equiv*) with its fitted geometry
  curve (*angular_error_fit*) and the normalised harmonics.

## Troubleshooting

- **Phase in the thousands of degrees, or random wiggles**: the tone frequency does not match the record.
  Choose the right record or enter the frequency by hand.
- **"Plotting vs TIME" warning**: the detected tone length does not match the profile. Wrong record, partial
  recording, or a strong speed error. Choose the record side explicitly to force radius, or raise the allowed
  span deviation.
- **"Tone not detected"**: raise *Tone detect: dB below 90th percentile* (a more tolerant threshold) or shorten
  the hold time. Also check that the recording contains the tone at a good level.
- **"Fit not applied"**: the fit needs a record profile (radius axis) and the fixed value of the chosen model
  (pivot-to-spindle distance or effective length).
- **Fit curve does not follow the data**: look at the smoothing, the trim, and the x-axis limits, and check
  that c and the pivot-to-spindle distance are right.
- **Analysis failed**: the message shown names the cause, for example a mono file or a silent file.

## Adding your own record

Add an entry to `RECORD_PROFILES` in `app.py` with a label, `tone_hz`, one or more sides (rpm, band duration in
seconds, start and end radius in mm) and optional marker radii. It then appears in the Record list.

*This app is experimental. Check the results against your own expectations before relying on them.*
"""


@st.dialog("User guide", width="large")
def show_guide():
    st.markdown(GUIDE_MD)


st.markdown("""<style>
/* make the zenith-offset setting stand out in the sidebar */
section[data-testid="stSidebar"] .st-key-cfix_on label, section[data-testid="stSidebar"] .st-key-cfix_on label *,
section[data-testid="stSidebar"] .st-key-cfix label, section[data-testid="stSidebar"] .st-key-cfix label *,
section[data-testid="stSidebar"] .st-key-alphaknown label, section[data-testid="stSidebar"] .st-key-alphaknown label * {
    color: #7fc8ff !important; font-weight: 600;
}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    if st.button("📖 User guide", width="stretch", key="open_guide"):
        show_guide()
    st.header("Settings")
    # Record and tone frequency sit OUTSIDE the form so they update immediately when the record is
    # changed: the frequency box then shows the record's real tone (Sperling 1000 Hz, clearaudio 3150 Hz)
    # and the record-side list refreshes without pressing Run.
    prof_choice = st.selectbox("Record", ["sperling", "clearaudio", "none (time axis)"], index=0, key="prof_choice")
    RECORD_PROFILE = None if prof_choice.startswith("none") else prof_choice
    _hz_default = (RECORD_PROFILES[RECORD_PROFILE].get("tone_hz") if RECORD_PROFILE else None) or D["FREQ_CENTER"]
    FREQ_CENTER = st.number_input("Tone frequency (Hz), peak mode", value=float(_hz_default), min_value=1.0,
                                  step=10.0, key=f"tone_hz_{prof_choice}",
                                  help="Defaults to the selected record's tone. Edit it to track a different frequency.")
    # Fit model is also outside the form: the fields below it (fixed pivot-to-spindle OR fixed effective
    # length OR the shape guesses) are shown or hidden immediately, so only the relevant ones appear.
    _fit_labels = {"geometry": "Geometry: fixed pivot-to-spindle",
                   "geometry_L": "Geometry: fixed effective length",
                   "shape": "Shape (free nulls)"}
    PHASE_FIT_MODEL = st.selectbox("Tracking-error fit model", list(_fit_labels), index=0,
                                   format_func=_fit_labels.get, key="fit_model",
                                   help="Fixed pivot-to-spindle: exact pivoted-arm model for an arm whose pivot-to-spindle "
                                        "distance is fixed; fits overhang (so effective length) and offset angle. "
                                        "Fixed effective length: for an arm with a fixed effective length and an "
                                        "adjustable base (e.g. SME); fits the pivot-to-spindle distance and offset angle. "
                                        "Shape: c + k(r-n1)(r-n2)/r with free nulls.")
    with st.form("settings"):
        # Submit button first, so it sits at the top of the form (it still submits every widget below it).
        submitted = st.form_submit_button("Run analysis", type="primary", width="stretch")
        with st.expander("Analysis", expanded=True):
            MODE = st.selectbox("Mode", ["peak", "sweep", "broadband"], index=0,
                                help="peak: track the tone near a set frequency. sweep: track the dominant bin in a band. "
                                     "broadband: magnitude-weighted phase over a band.")
            FREQ_TOL_HZ = st.number_input("Search window ± (Hz), peak mode", value=D["FREQ_TOL_HZ"], step=1.0)
            c1, c2 = st.columns(2)
            sw_lo = c1.number_input("Sweep band low (Hz)", value=D["SWEEP_BAND"][0])
            sw_hi = c2.number_input("Sweep band high (Hz)", value=D["SWEEP_BAND"][1])
            b_lo = c1.number_input("Broadband low (Hz)", value=D["FREQ_BAND"][0])
            b_hi = c2.number_input("Broadband high (Hz)", value=D["FREQ_BAND"][1])
            NFFT = st.selectbox("FFT size (NFFT)", [4096, 8192, 16384, 32768, 65536], index=3)
            HOP = st.selectbox("Hop size", [512, 1024, 2048, 4096, 8192], index=2)
            SMOOTH_MS = st.number_input("Smoothing (ms), 0 = off", value=D["SMOOTH_MS"], min_value=0.0, step=1000.0)
            PHASE_UNIT = st.selectbox("Phase unit", ["degrees", "radians"], index=0)
            UNWRAP_PHASE = st.checkbox("Unwrap phase", value=D["UNWRAP_PHASE"])
            TITLE_TEXT = st.text_input("Title text (added to plot titles)", value=D["TITLE_TEXT"])
            ylim_on = st.checkbox("Fix phase y-axis limits", value=False)
            c1, c2 = st.columns(2)
            y_lo = c1.number_input("Phase y min", value=-3.0)
            y_hi = c2.number_input("Phase y max", value=3.0)

        with st.expander("Record profile", expanded=True):
            side_opts = ["auto"] + (list(RECORD_PROFILES[RECORD_PROFILE]["sides"]) if RECORD_PROFILE else [])
            RECORD_SIDE = st.selectbox("Record side (rpm)", side_opts, index=0, key=f"side_{prof_choice}",
                                       help="auto picks the side whose band length matches the detected tone.")
            RADIUS_ALIGN = st.selectbox("Radius alignment", ["both", "onset", "manual"], index=0,
                                        help="both: detect tone start and end and stretch to fit (absorbs speed error). "
                                             "onset: detect start, use nominal duration. manual: start at the offset below.")
            RADIUS_T_OFFSET_S = st.number_input("Manual tone start (s)", value=D["RADIUS_T_OFFSET_S"], min_value=0.0)
            c1, c2 = st.columns(2)
            xl_lo = c1.number_input("X-axis min (mm)", value=D["RADIUS_XLIM_MM"][0])
            xl_hi = c2.number_input("X-axis max (mm)", value=D["RADIUS_XLIM_MM"][1])
            RADIUS_INVERT_X = st.checkbox("Outer radius on the left (playback order)", value=D["RADIUS_INVERT_X"])
            RADIUS_MAX_SPAN_DEV = st.number_input("Allowed span deviation (%)", value=D["RADIUS_MAX_SPAN_DEV"] * 100, step=1.0) / 100.0
            RADIUS_DETECT_DROP_DB = st.number_input("Tone detect: dB below 90th percentile", value=D["RADIUS_DETECT_DROP_DB"])
            RADIUS_DETECT_HOLD_S = st.number_input("Tone detect: hold time (s)", value=D["RADIUS_DETECT_HOLD_S"])

        with st.expander("2nd harmonic", expanded=False):
            HARM_RADIUS_NORM = st.checkbox("Normalise H2 to radius (% × r / r_ref)", value=D["HARM_RADIUS_NORM"])
            HARM_NORM_REF_MM = opt_float("Reference radius (mm)", D["HARM_NORM_REF_MM"], "hnref",
                                         help="Unchecked = the profile's outer radius (146.05 mm for Sperling).")
            HARM_NORM_SHOW_RAW = st.checkbox("Also show un-normalised H2 (dashed)", value=D["HARM_NORM_SHOW_RAW"])

        with st.expander("Tracking-error fit", expanded=True):
            PHASE_FIT_TRACKING_ERROR = st.checkbox("Fit tracking-error curve", value=D["PHASE_FIT_TRACKING_ERROR"])
            st.caption(f"Model: **{_fit_labels[PHASE_FIT_MODEL]}** (change it at the top of the sidebar)")
            # Show only the fields that the chosen model uses; the others keep their defaults and are not used.
            PHASE_FIT_PIVOT_SPINDLE_MM = D["PHASE_FIT_PIVOT_SPINDLE_MM"]
            PHASE_FIT_EFF_LENGTH_MM = D["PHASE_FIT_EFF_LENGTH_MM"]
            PHASE_FIT_OVERHANG_GUESS_MM = D["PHASE_FIT_OVERHANG_GUESS_MM"]
            PHASE_FIT_OFFSET_ANGLE_GUESS_DEG = D["PHASE_FIT_OFFSET_ANGLE_GUESS_DEG"]
            PHASE_FIT_N1_GUESS_MM = D["PHASE_FIT_N1_GUESS_MM"]
            PHASE_FIT_N2_GUESS_MM = D["PHASE_FIT_N2_GUESS_MM"]
            if PHASE_FIT_MODEL == "geometry":
                PHASE_FIT_PIVOT_SPINDLE_MM = st.number_input("Pivot-to-spindle (mm), fixed",
                                                             value=D["PHASE_FIT_PIVOT_SPINDLE_MM"], format="%.2f",
                                                             help="Your arm's measured pivot-to-spindle distance. Not fitted.")
            elif PHASE_FIT_MODEL == "geometry_L":
                PHASE_FIT_EFF_LENGTH_MM = st.number_input("Effective length (mm), fixed",
                                                          value=D["PHASE_FIT_EFF_LENGTH_MM"], format="%.2f",
                                                          help="Pivot to stylus distance of your arm, from its "
                                                               "specification. Not fitted; the pivot-to-spindle "
                                                               "distance is fitted instead.")
            if PHASE_FIT_MODEL in ("geometry", "geometry_L"):
                PHASE_FIT_OVERHANG_GUESS_MM = st.number_input("Overhang start guess (mm)", value=D["PHASE_FIT_OVERHANG_GUESS_MM"], format="%.2f")
                PHASE_FIT_OFFSET_ANGLE_GUESS_DEG = st.number_input("Offset angle start guess (deg)", value=D["PHASE_FIT_OFFSET_ANGLE_GUESS_DEG"], format="%.2f")
                PHASE_FIT_ALPHA_KNOWN = st.checkbox("Offset angle is known: fix it and fit the zenith offset",
                                                    value=D["PHASE_FIT_ALPHA_KNOWN"], key="alphaknown",
                                                    help="Holds the arm offset angle at the value above and fits the constant "
                                                         "angular (zenith) offset instead. Offset angle and zenith offset are "
                                                         "degenerate, so only one of them can be fitted. Needs the angular-error "
                                                         "conversion; the manual zenith offset is then ignored.")
            else:
                c1, c2 = st.columns(2)
                PHASE_FIT_N1_GUESS_MM = c1.number_input("Inner null guess (mm), shape", value=D["PHASE_FIT_N1_GUESS_MM"])
                PHASE_FIT_N2_GUESS_MM = c2.number_input("Outer null guess (mm), shape", value=D["PHASE_FIT_N2_GUESS_MM"])
            PHASE_FIT_OFFSET_DEG = opt_float("Known constant angular offset (zenith), in phase deg at r_ref", D["PHASE_FIT_OFFSET_DEG"], "cfix",
                                             help="With the angular-error conversion on this is a constant ANGULAR error (cartridge zenith / record tilt), "
                                                  "in degrees of phase at the reference radius (zenith in degrees = value / k); it gives a phase that "
                                                  "varies as 1/r. With the conversion off it is a constant phase offset (R15). "
                                                  "Checked: fixed to this value, making offset angle and nulls identifiable. "
                                                  "Unchecked: c is fitted, and nulls are reported as indeterminate (geometry) "
                                                  "or approximate (shape).")
            PHASE_FIT_TRIM_S = st.number_input("Trim each end before fitting (s)", value=D["PHASE_FIT_TRIM_S"], min_value=0.0)
            PHASE_FIT_SHOW_OPTIMAL = st.checkbox("Show optimal (Löfgren A) curve", value=D["PHASE_FIT_SHOW_OPTIMAL"])
            PHASE_FIT_SHOW_ZENITH_CURVE = st.checkbox("Also show optimal curve shifted by the zenith angular offset",
                                                      value=D["PHASE_FIT_SHOW_ZENITH_CURVE"],
                                                      help="Pure theory curve plus the constant angular (zenith) offset used by the fit. "
                                                           "Needs the angular-error conversion and the optimal curve.")
            PHASE_FIT_LOFGREN_R1_MM = opt_float("Löfgren inner radius R1 (mm)", D["PHASE_FIT_LOFGREN_R1_MM"], "lgr1",
                                                help="Unchecked = the record band's inner radius.")
            PHASE_FIT_LOFGREN_R2_MM = opt_float("Löfgren outer radius R2 (mm)", D["PHASE_FIT_LOFGREN_R2_MM"], "lgr2",
                                                help="Unchecked = the record band's outer radius.")

        with st.expander("Angular error (phase to angle conversion)", expanded=True):
            ANG_NORM = st.checkbox("Convert phase to angular error: phase × (r / r_ref)^p − offset", value=D["ANG_NORM"],
                                   help="At a constant angular error the phase grows toward the centre (lower groove "
                                        "speed, shorter wavelength). This plots the angular error on page 2 and makes "
                                        "the geometry fits use the same response. Untick for R15 behaviour.")
            ANG_REF_MM = opt_float("Reference radius r_ref (mm)", D["ANG_REF_MM"], "angref",
                                   help="Unchecked = the record band's outer radius. The angular error is given in "
                                        "degrees of phase this angular error would produce at r_ref.")
            ANG_EXPONENT = st.number_input("Exponent p", value=float(D["ANG_EXPONENT"]), step=0.1, format="%.2f",
                                           help="1 = phase per degree of angular error proportional to 1/r (groove "
                                                "speed). 0 = no conversion (phase proportional to angle).")

        SHOW_PAGE2 = st.checkbox("Show page 2 (phase + angular error + H2 + fit table)", value=D["SHOW_PAGE2"])

cfg = dict(
    MODE=MODE, FREQ_CENTER=FREQ_CENTER, FREQ_TOL_HZ=FREQ_TOL_HZ,
    SWEEP_BAND=(sw_lo, sw_hi), FREQ_BAND=(b_lo, b_hi),
    NFFT=NFFT, HOP=HOP, PHASE_UNIT=PHASE_UNIT, UNWRAP_PHASE=UNWRAP_PHASE, SMOOTH_MS=SMOOTH_MS,
    TITLE_TEXT=TITLE_TEXT, Y_AXIS_LIMITS=(y_lo, y_hi) if ylim_on else None,
    RECORD_PROFILE=RECORD_PROFILE, RECORD_SIDE=RECORD_SIDE, RADIUS_ALIGN=RADIUS_ALIGN,
    RADIUS_T_OFFSET_S=RADIUS_T_OFFSET_S, RADIUS_MAX_SPAN_DEV=RADIUS_MAX_SPAN_DEV,
    RADIUS_DETECT_DROP_DB=RADIUS_DETECT_DROP_DB, RADIUS_DETECT_HOLD_S=RADIUS_DETECT_HOLD_S,
    RADIUS_ENV_BLOCK_MS=D["RADIUS_ENV_BLOCK_MS"], RADIUS_INVERT_X=RADIUS_INVERT_X,
    RADIUS_XLIM_MM=(xl_lo, xl_hi),
    HARM_RADIUS_NORM=HARM_RADIUS_NORM, HARM_NORM_REF_MM=HARM_NORM_REF_MM, HARM_NORM_SHOW_RAW=HARM_NORM_SHOW_RAW,
    SHOW_PAGE2=SHOW_PAGE2,
    PHASE_FIT_TRACKING_ERROR=PHASE_FIT_TRACKING_ERROR, PHASE_FIT_MODEL=PHASE_FIT_MODEL,
    PHASE_FIT_N1_GUESS_MM=PHASE_FIT_N1_GUESS_MM, PHASE_FIT_N2_GUESS_MM=PHASE_FIT_N2_GUESS_MM,
    PHASE_FIT_PIVOT_SPINDLE_MM=PHASE_FIT_PIVOT_SPINDLE_MM, PHASE_FIT_EFF_LENGTH_MM=PHASE_FIT_EFF_LENGTH_MM,
    PHASE_FIT_OVERHANG_GUESS_MM=PHASE_FIT_OVERHANG_GUESS_MM,
    PHASE_FIT_OFFSET_ANGLE_GUESS_DEG=PHASE_FIT_OFFSET_ANGLE_GUESS_DEG, PHASE_FIT_TRIM_S=PHASE_FIT_TRIM_S,
    PHASE_FIT_ALPHA_KNOWN=(PHASE_FIT_ALPHA_KNOWN if PHASE_FIT_MODEL in ("geometry", "geometry_L") else False), PHASE_FIT_SHOW_ZENITH_CURVE=PHASE_FIT_SHOW_ZENITH_CURVE, PHASE_FIT_SHOW_OPTIMAL=PHASE_FIT_SHOW_OPTIMAL, PHASE_FIT_LOFGREN_R1_MM=PHASE_FIT_LOFGREN_R1_MM,
    PHASE_FIT_LOFGREN_R2_MM=PHASE_FIT_LOFGREN_R2_MM, PHASE_FIT_OFFSET_DEG=PHASE_FIT_OFFSET_DEG,
    ANG_NORM=ANG_NORM, ANG_REF_MM=ANG_REF_MM, ANG_EXPONENT=ANG_EXPONENT,
)

up = st.file_uploader("Stereo WAV or FLAC (FLAC is about half the size and uploads faster)",
                      type=["wav", "flac"])

if submitted:
    if up is None:
        st.warning("Upload a stereo WAV or FLAC file first, then press Run analysis.")
    else:
        suffix = Path(up.name).suffix.lower() or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            up.seek(0)
            while True:
                chunk = up.read(8 * 1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp.close()
            bar = st.progress(0.0, text="Analysing…")
            R = run_analysis(tmp.name, up.name, cfg,
                             progress=lambda p: bar.progress(p, text=f"Analysing… {p*100:.0f} %"))
            bar.empty()
            pngs = {"page1": fig_to_png(fig_page1(R, cfg))}
            if cfg["SHOW_PAGE2"]:
                pngs["page2"] = fig_to_png(fig_page2(R, cfg))
            if R["MODE"] in ("peak", "sweep") and np.any(~np.isnan(R["tracked_freqs"])):
                pngs["freq"] = fig_to_png(fig_freq(R, cfg))
            st.session_state["result"] = dict(
                name=up.name, log=R["log"], pngs=pngs, csv=R["csv_bytes"], table=R["phase_fit_table"],
                fit_note=R["phase_fit_note"], radius_note=R["radius_note"],
            )
            del R
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Analysis failed: {e}")
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            plt.close('all')
            gc.collect()

res = st.session_state.get("result")
if res:
    st.subheader(f"{SCRIPT_NAME} {SCRIPT_REV}  ·  {res['name']}")
    for level, txt in res["log"]:
        if level == "warn":
            st.warning(txt)
    if res["table"]:
        has_opt = any(o is not None for _, _, o in res["table"])
        tdf = pd.DataFrame(
            {"Fitted": [f for _, f, _ in res["table"]],
             **({"Optimal (target)": [o if o is not None else "—" for _, _, o in res["table"]]} if has_opt else {})},
            index=[l for l, _, _ in res["table"]])
        st.dataframe(tdf, width="content")

    tabs = st.tabs(["Page 1: phase, delay, level, H2"]
                   + (["Page 2: phase, angular error, H2, fit"] if "page2" in res["pngs"] else [])
                   + (["Tracked frequency"] if "freq" in res["pngs"] else []))
    keys = ["page1"] + (["page2"] if "page2" in res["pngs"] else []) + (["freq"] if "freq" in res["pngs"] else [])
    base = Path(res["name"]).stem
    for tab, k in zip(tabs, keys):
        with tab:
            st.image(res["pngs"][k], width="stretch")
            st.download_button("Download PNG", res["pngs"][k], file_name=f"{base}_{k}.png", mime="image/png", key=f"dl_{k}")

    st.download_button("Download time series CSV", res["csv"], file_name=f"{base}_timeseries.csv", mime="text/csv")
    with st.expander("Processing log"):
        st.code("\n".join(t for _, t in res["log"]), language=None)
else:
    st.info("Upload a recording, check the settings in the sidebar, and press **Run analysis**. "
            "Defaults match the R15 script: Sperling profile, peak mode at 1 kHz, geometry fit with pivot-to-spindle 215.35 mm.")

st.caption(f"{SCRIPT_NAME} {SCRIPT_REV} · Balle-Clorin · PolyForm Noncommercial 1.0.0")
