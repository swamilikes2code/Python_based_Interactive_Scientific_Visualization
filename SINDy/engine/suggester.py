# engine/ai_suggester.py
#
# PURPOSE
# -------
# Pure data-analysis heuristic that recommends starting SINDy hyperparameters
# (library / polynomial degree / sparsity threshold) for a given dataset —
# no UI/Bokeh dependency, so it can be unit-tested or reused outside the
# Train tab (e.g. batch processing, a future API).

import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



def analyze_data_linearity(df):
    """
    Analyze an uploaded/selected dataset and suggest SINDy hyperparameters.

    Strategy
    --------
    1. Estimate dX/dt via Savitzky-Golay smoothing + finite differences
       (smoothing reduces the amplification of noise inherent to
       numerical differentiation).
    2. Fit polynomial regressions of degree 1, 2, and 3 from X -> dX/dt
       and compare their R² scores. The *smallest* degree that gives a
       meaningfully better fit than the previous degree is selected —
       this avoids over-suggesting complexity when a simpler model
       already explains the dynamics well (Occam's razor).
    3. Run an FFT-based periodicity check (dominant peak vs mean
       spectral energy) — currently informational only; see the note
       in Step 6 for why the library suggestion itself is not directly
       applied to the UI.
    4. Estimate the noise floor from the high-frequency tail of the FFT
       and map it to a suggested sparsity threshold.

    Returns
    -------
    tuple(str, int, float, str)
        (suggested_library, suggested_degree, suggested_threshold, reason_text)
        reason_text is a human-readable explanation shown in the UI.
    """
    try:
        t = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
        n_vars = X.shape[1]
        dt = np.mean(np.diff(t)) if len(t) > 1 else 0.1

        # ── 1. Calculate derivative ──────────────────────────────
        # Savitzky-Golay smoothing before differentiating: this is
        # critical because raw finite differences amplify sensor/CSV
        # noise, which would otherwise bias the linearity test below.
        dXdt = np.zeros_like(X)
        window = min(11, len(t) // 10 * 2 + 1)
        window = max(window, 5)
        for i in range(n_vars):
            smoothed = savgol_filter(
                X[:, i], window_length=window, polyorder=3)
            dXdt[:, i] = np.gradient(smoothed, dt)

        # ── 2. Compare R² of degree 1, 2, 3 ────────────────────────
        # Fit an ordinary polynomial regression (not SINDy/STLSQ) purely
        # as a fast proxy to gauge how nonlinear the system "looks".
        r2_scores = {}
        for deg in [1, 2, 3]:
            poly = PolynomialFeatures(degree=deg, include_bias=True)
            X_poly = poly.fit_transform(X)
            lr = LinearRegression(fit_intercept=False).fit(X_poly, dXdt)
            r2_scores[deg] = r2_score(dXdt, lr.predict(X_poly),
                                      multioutput='uniform_average')

        r2_linear = r2_scores[1]
        r2_deg2 = r2_scores[2]
        r2_deg3 = r2_scores[3]

        # ── 3. Choose minimal degree that satisfies R² ──────────────────
        # If degree=1 is already sufficient → treat as a linear system.
        if r2_linear >= 0.85:
            sug_degree = 1
        # If degree=2 meaningfully improves over degree=1 → nonlinear (quadratic-ish).
        elif r2_deg2 - r2_linear >= 0.05:
            sug_degree = 2
        # If degree=3 meaningfully improves over degree=2 → higher-order nonlinearity.
        elif r2_deg3 - r2_deg2 >= 0.05:
            sug_degree = 3
        # No degree gives a meaningful improvement → default back to linear
        # (a weakly-nonlinear system may simply be indistinguishable from
        # noise at this sampling rate/noise level).
        else:
            sug_degree = 1

        # ── 4. FFT — detect periodicity ──────────────────────────────
        # A single dominant peak that is >5x the mean spectral amplitude
        # indicates a strongly oscillatory/periodic signal.
        is_periodic = False
        for i in range(n_vars):
            # remove DC offset before FFT
            signal = X[:, i] - np.mean(X[:, i])
            fft_vals = np.abs(np.fft.rfft(signal))
            peaks = fft_vals[1:]  # skip the DC bin
            if len(peaks) > 0:
                if np.max(peaks) > 5 * np.mean(peaks):
                    is_periodic = True
                    break

        # ── 5. Noise estimate → suggest threshold ────────────────────
        # Approximate the noise floor as the median amplitude of the
        # top 20% highest frequencies (where genuine physical signal
        # content is usually negligible for smooth trajectories).
        noise_estimates = []
        for i in range(n_vars):
            amp = np.abs(np.fft.rfft(X[:, i])) / len(t)
            high = np.sort(amp)[-max(1, int(len(amp) * 0.2)):]
            noise_estimates.append(float(np.median(high)))
        noise_level = float(np.mean(noise_estimates))

        if noise_level < 0.01:
            sug_threshold = 0.05
        elif noise_level < 0.05:
            sug_threshold = 0.10
        else:
            sug_threshold = 0.20

        # ── 6. Library suggestion (informational only — NOT applied to UI) ──
        # NOTE: Earlier testing (see project history) showed that Fourier/
        # Combined suggestions frequently misfire on purely polynomial
        # systems that merely *look* oscillatory (e.g. coupled spring-mass),
        # because the peak-ratio heuristic can't distinguish "oscillatory
        # data" from "equations that actually contain sin/cos terms".
        # We therefore compute sug_library for display/reasoning purposes
        # only; apply_suggestion() in train_tab.py deliberately does NOT
        # set library_select.value from this result. Only degree and
        # threshold are auto-applied.
        if is_periodic and r2_linear < 0.92:
            sug_library = "Combined"
        elif is_periodic and r2_linear >= 0.92:
            sug_library = "Fourier"
        else:
            sug_library = "Polynomial"

        reason = f"Degree: {sug_degree}; Threshold: {sug_threshold}; Noise ≈ {noise_level:.4f}."
        return sug_library, sug_degree, sug_threshold, reason

    except Exception as e:
        # Fail-safe defaults so a bad/edge-case CSV never blocks the user
        # from proceeding to manual configuration.
        return "Polynomial", 1, 0.10, f"Error analyzing data: {e}"