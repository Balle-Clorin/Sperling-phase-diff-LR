# Sperling Phase Alignment Analyser

Streamlit web app that measures the **R-L phase difference, time delay, level and 2nd harmonic** of a stereo needle-drop of a single-tone test record, maps time to **groove radius**, and fits a **tracking-error curve** to check tonearm alignment.

Web version of `ANYSIGNAL_-phasediff_2harm_sweep__R15.py`. The analysis code is R15 unchanged; the settings that were constants at the top of the script are in the sidebar.

## What it does

- STFT phase tracking in three modes: **peak** (tone near a set frequency, default 1 kHz), **sweep** (dominant bin in a band) and **broadband** (magnitude-weighted phase over a band).
- Record profiles for **Sperling** (33 and 45 rpm sides, 146.05 → 57.5 mm) and **Clearaudio** (127 → 69 mm). Tone start and end are detected automatically and mapped linearly to the band's outer and inner radius, which also gives the speed error.
- 2nd harmonic per channel, optionally normalised to radius (% × r / r_ref).
- Tracking-error fit:
  - **geometry**: exact pivoted-arm model. Fits overhang and offset angle for your pivot-to-spindle distance and derives the null radii from that geometry.
  - **shape**: c + k(r − n1)(r − n2)/r with free nulls.
  - Optional Löfgren A comparison curve at the same pivot-to-spindle distance.
- With a free phase offset, offset angle and nulls are not identifiable from the curve alone. Set a known constant offset to resolve them.

## Output

- Page 1: phase, delay, level and H2 vs radius (or time).
- Page 2: phase and H2 with the fit table (fitted vs optimal).
- Tracked frequency plot.
- PNG downloads and a time-series CSV with all columns from R15.

## Input

Stereo WAV or FLAC. FLAC is about half the size and uploads faster. The upload limit is set to 1000 MB in `.streamlit/config.toml`.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## License

PolyForm Noncommercial 1.0.0. See `LICENSE`.
