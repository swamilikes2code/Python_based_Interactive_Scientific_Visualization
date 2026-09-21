# engine/sindy_model.py

import pysindy as ps
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class SINDyEngine:
    def __init__(self):
        self.model = None
        self.feature_names = []

    # ------------------------------------------------------------------
    # Helper: build library
    # ------------------------------------------------------------------
    def _build_library(self, lib_type, poly_degree):
        if lib_type == "Polynomial":
            return ps.PolynomialLibrary(degree=int(poly_degree))
        elif lib_type == "Fourier":
            return ps.FourierLibrary(n_frequencies=int(poly_degree))
        elif lib_type == "Combined":
            return (ps.PolynomialLibrary(degree=int(poly_degree))
                    + ps.FourierLibrary(n_frequencies=int(poly_degree)))
        else:
            raise ValueError(
                f"Inappropriate library type: '{lib_type}'. "
                "Choose: Polynomial, Fourier, Combined."
            )

    # ------------------------------------------------------------------
    # Calculate derivatives on continuous timespan
    # ------------------------------------------------------------------
    def compute_derivatives(self, X, t):
        """
        Calculate dx/dt on all data using SmoothedFiniteDifference.
        Return dX shape (n_samples, n_features). Requires t to be
        continuous/increasing for the whole array — never call this on a
        pooled multi-trajectory array where t restarts at 0 partway
        through (see _pool_trajectories for the correct way to handle
        multiple trajectories).
        """
        diff = ps.SmoothedFiniteDifference()
        dX = diff(X, t=t)
        return dX

    # ------------------------------------------------------------------
    # Helper: pool multiple trajectories of the SAME system
    # ------------------------------------------------------------------
    def _pool_trajectories(self, trajectories):
        """
        trajectories : list of (X_i, t_i) — separate trajectories of the
            same system (e.g. different initial conditions).

        IMPORTANT: derivatives are computed PER TRAJECTORY, because
        SmoothedFiniteDifference needs a continuous time axis. Concatenating
        the CSVs first and differentiating once would treat every file
        boundary (where t restarts at 0) as a physical jump, producing
        huge spurious derivatives at the seams.

        Returns (X_pool, dX_pool, t_pool): row-wise stacks of all
        per-trajectory arrays. t_pool is only used for plotting/diagnostics
        — it is NEVER fed back into compute_derivatives as a whole.
        """
        X_parts, dX_parts, t_parts = [], [], []
        for X_i, t_i in trajectories:
            X_i = np.asarray(X_i, dtype=float)
            t_i = np.asarray(t_i, dtype=float)
            X_parts.append(X_i)
            dX_parts.append(self.compute_derivatives(X_i, t_i))
            t_parts.append(t_i)
        return np.vstack(X_parts), np.vstack(dX_parts), np.concatenate(t_parts)

    # ------------------------------------------------------------------
    # Fit model with random split on (X, dX) pairs
    # ------------------------------------------------------------------
    def fit_model(self, X, t, poly_degree, threshold, names,
                  lib_type="Polynomial",
                  train_frac=0.6, random_seed=42, split_method="random",
                  extra_trajectories=None):
        """
        Approach:
          1. Calculate dX PER TRAJECTORY (each file has its own continuous
             time axis), then pool all (X, dX) pairs into one big design set.
          2. Split the pooled (X, dX) pairs into train/val, using one of
             3 strategies (see below).
          3. Fit SINDy on the train split.
          4. Validate on the val split (compare dX_pred vs dX_val).

        extra_trajectories : optional list of (X_i, t_i) tuples — extra
            trajectories of the SAME system measured from DIFFERENT initial
            conditions. Pooling them means the fitted coefficients are
            constrained by every initial condition at once -> a more
            robust, globally-valid model instead of one tuned to a single
            trajectory. Pass None (or omit) for the plain single-trajectory
            case — behavior is then identical to a single-file fit.

        Returns: (model, train_idx, val_idx, metrics_train, metrics_val,
                  X_pool, t_pool)
            train_idx/val_idx index into the POOLED arrays X_pool/t_pool,
            so plotting code can scatter train/val points directly.
        """
        self.feature_names = names

        # Step 1: per-trajectory derivatives, then pool.
        traj_list = [(np.asarray(X, dtype=float), np.asarray(t, dtype=float))]
        for Xi, ti in (extra_trajectories or []):
            traj_list.append((np.asarray(Xi, dtype=float),
                              np.asarray(ti, dtype=float)))
        X_pool, dX_pool, t_pool = self._pool_trajectories(traj_list)

        # CHANGED: per-trajectory sample counts, needed so "random block"
        # and "time-based" splits below can respect trajectory boundaries
        # instead of treating the pooled array as one seamless timeline.
        lengths = [len(ti) for _, ti in traj_list]

        # The split below operates on the pooled sample index space [0, n).
        n = len(t_pool)
        n_train = int(n * train_frac)
        rng = np.random.default_rng(random_seed)

        if split_method == "random sampling":
            # Random point-level split over the POOLED pairs — every
            # initial condition contributes to both train and val.
            indices = rng.permutation(n)

            train_idx = np.sort(indices[:n_train])
            val_idx = np.sort(indices[n_train:])
            # np.sort() restores chronological order within each subset,
            # so downstream code that assumes increasing time (plotting a
            # line, or solve_ivp) never breaks.

        elif split_method == "time-based":
            # CHANGED: chronological split done INDEPENDENTLY per
            # trajectory, then concatenated — instead of slicing the
            # pooled array as one block. With multi-IC data, slicing the
            # pool directly would make "first n_train samples" equal to
            # "the first few uploaded files in full", biasing validation
            # coverage by upload order. Splitting per-trajectory means
            # every initial condition contributes its own train_frac% of
            # early time to train, and the remaining tail to val.
            train_parts, val_parts = [], []
            offset = 0
            for length in lengths:
                n_train_i = int(length * train_frac)
                train_parts.append(np.arange(offset, offset + n_train_i))
                val_parts.append(np.arange(offset + n_train_i, offset + length))
                offset += length
            train_idx = np.concatenate(train_parts)
            val_idx = np.concatenate(val_parts)

        elif split_method == "random block":
            # CHANGED: blocks are now built per-trajectory (via
            # _make_blocks_multi) so a single block can never splice
            # together timesteps from two physically unrelated initial
            # conditions — which would silently defeat the purpose of a
            # block split (preserving local temporal autocorrelation)
            # right at every file boundary.
            blocks = self._make_blocks_multi(lengths)
            n_blocks_total = len(blocks)
            block_ids = rng.permutation(n_blocks_total)
            n_train_blocks = int(n_blocks_total * train_frac)

            train_blocks = sorted(block_ids[:n_train_blocks])
            val_blocks = sorted(block_ids[n_train_blocks:])

            train_idx = np.sort(np.concatenate([blocks[b] for b in train_blocks]))
            val_idx = np.sort(np.concatenate([blocks[b] for b in val_blocks]))

        else:
            raise ValueError(
                f"Unknown split_method '{split_method}'. "
                "Use 'random sampling', 'time-based' or 'random block'."
            )

        X_train = X_pool[train_idx]
        dX_train = dX_pool[train_idx]
        X_val = X_pool[val_idx]
        dX_val = dX_pool[val_idx]

        # Step 3: fit SINDy with (X_train, dX_train)
        library = self._build_library(lib_type, poly_degree)
        optimizer = ps.STLSQ(threshold=threshold)
        self.model = ps.SINDy(
            optimizer=optimizer,
            feature_library=library,
            differentiation_method=ps.FiniteDifference()  # dummy, x_dot supplied directly
        )

        t_dummy = np.arange(len(train_idx), dtype=float)
        self.model.fit(X_train, t=t_dummy,
               x_dot=dX_train, feature_names=names)

        # Step 4: calculate metrics on derivative space
        dX_train_pred = self.model.predict(X_train)
        dX_val_pred = self.model.predict(X_val)

        metrics_train = self._metrics_on_dx(dX_train, dX_train_pred)
        metrics_val = self._metrics_on_dx(dX_val,   dX_val_pred)

        return (self.model, train_idx, val_idx,
                metrics_train, metrics_val, X_pool, t_pool)

    # ------------------------------------------------------------------
    # Metrics on dx/dt space
    # ------------------------------------------------------------------
    def _metrics_on_dx(self, dX_true, dX_pred):
        mse = mean_squared_error(dX_true, dX_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(dX_true, dX_pred))
        r2 = float(r2_score(dX_true, dX_pred, multioutput='uniform_average'))
        residual = dX_true - dX_pred
        rss = np.sum(residual**2)
        return {'mse': float(mse), 'rmse': rmse, 'mae': mae, 'r2': r2, 'rss': rss}

    # ------------------------------------------------------------------
    # Equations, simulate, metrics on x(t)
    # ------------------------------------------------------------------
    def get_equations(self, precision=3):
        if self.model is None:
            print("Warning: Model is not fitted.")
            return []

        names = self.feature_names if self.feature_names else [
            f"x{i}" for i in range(len(self.model.equations()))]

        rhs_list = self.model.equations(precision=precision)

        full_equations = []
        for i, rhs in enumerate(rhs_list):
            lhs = f"d({names[i]})/dt"
            full_equations.append(f"{lhs} = {rhs}")

        return full_equations

    def simulate(self, x0, t_range):
        if self.model is None:
            print("Warning: Model is not fitted.")
            return None
        try:
            result = self.model.simulate(x0, t_range)
        except Exception as e:
            raise RuntimeError(f"Simulate failed: {e}")
        if np.any(np.isinf(result)) or np.any(np.isnan(result)):
            raise RuntimeError(
                "Simulation diverged (overflow/nan). "
                "Try increase Sparsity Threshold or decrease Degree."
            )
        return result

    def simulate_with_model(self, model_instance, x0, t):
        try:
            return model_instance.simulate(x0, t)
        except Exception as e:
            print(f"Error when simulate using old model: {e}")
            return np.zeros((len(t), len(x0)))

    def calculate_metrics(self, X_true, X_pred):
        """Metrics on x(t) — use for Test tab."""
        if np.any(np.isinf(X_pred)) or np.any(np.isnan(X_pred)):
            raise ValueError("X_pred contains inf or nan.")
        mse = mean_squared_error(X_true, X_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(X_true, X_pred))
        r2 = float(r2_score(X_true, X_pred, multioutput='uniform_average'))
        return {'mse': float(mse), 'rmse': rmse, 'mae': mae, 'r2': r2}

    def compute_diagnostics(self, X, t):
        """
        Compute raw diagnostic data for residual analysis after a SINDy fit.
        Single-trajectory version; see compute_diagnostics_multi() for the
        multi-initial-condition version.

        Returns a dict with:
            't'           : time array (shared x-axis for Plot 1)
            'residuals'   : dict {var_name: residual array}  -> Plot 1
            'fft_freqs'   : frequency array (shared x-axis for Plot 2)
            'fft_amps'    : dict {var_name: FFT amplitude}   -> Plot 2
            'dX_true'     : dict {var_name: true derivative} -> Plot 3
            'dX_pred'     : dict {var_name: pred derivative} -> Plot 3
            'stats'       : dict {var_name: {snr_db, autocorr, r2_dx}}
        """
        if self.model is None:
            return None

        dX_true = self.compute_derivatives(X, t)
        dX_pred = self.model.predict(X)
        residual = dX_true - dX_pred  # shape: (n_samples, n_features)

        n = len(t)
        dt = float(np.mean(np.diff(t)))

        fft_freqs = np.fft.rfftfreq(n, d=dt)

        result = {
            't':         t,
            'residuals': {},
            'fft_freqs': fft_freqs,
            'fft_amps':  {},
            'dX_true':   {},
            'dX_pred':   {},
            'stats':     {},
        }

        for i in range(residual.shape[1]):
            r = residual[:, i]
            name = self.feature_names[i] if self.feature_names else f"x{i}"

            result['residuals'][name] = r

            fft_amp = np.abs(np.fft.rfft(r)) / n
            result['fft_amps'][name] = fft_amp

            result['dX_true'][name] = dX_true[:, i]
            result['dX_pred'][name] = dX_pred[:, i]

            signal_power = np.var(dX_true[:, i])
            noise_power = np.var(r)
            r2_dx = float(
                r2_score(dX_true[:, i], dX_pred[:, i])) if signal_power > 0 else 0.0
            snr_db = 10 * np.log10(signal_power /
                                   noise_power) if noise_power > 0 else 99.0
            r_norm = r - r.mean()
            autocorr = float(np.corrcoef(r_norm[:-1], r_norm[1:])[0, 1])

            result['stats'][name] = {
                'r2_dx':    round(r2_dx,   3),
                'snr_db':   round(snr_db,  2),
                'autocorr': round(autocorr, 3),
            }

        return result

    # ------------------------------------------------------------------
    # Multi-trajectory diagnostics: same output structure as
    # compute_diagnostics(), but aggregated over several trajectories
    # (different initial conditions of the same fitted model):
    #   - residual-vs-time uses each trajectory's OWN t (no fake seams)
    #   - FFT = average of per-trajectory amplitude spectra, so a peak
    #     must be consistently present across ICs to stand out
    #   - scatter & stats use all pooled samples
    # ------------------------------------------------------------------
    def compute_diagnostics_multi(self, trajectories):
        if self.model is None:
            return None
        if len(trajectories) == 1:
            X0, t0 = trajectories[0]
            return self.compute_diagnostics(np.asarray(X0, float),
                                            np.asarray(t0, float))

        t_all, resid_all, dxt_all, dxp_all = [], [], [], []
        spectra = []
        fft_freqs = None
        for X_i, t_i in trajectories:
            X_i = np.asarray(X_i, dtype=float)
            t_i = np.asarray(t_i, dtype=float)
            dX_true = self.compute_derivatives(X_i, t_i)
            dX_pred = self.model.predict(X_i)
            r = dX_true - dX_pred

            t_all.append(t_i)
            resid_all.append(r)
            dxt_all.append(dX_true)
            dxp_all.append(dX_pred)

            n = len(t_i)
            dt = float(np.mean(np.diff(t_i)))
            freqs = np.fft.rfftfreq(n, d=dt)
            amp = np.abs(np.fft.rfft(r, axis=0)) / n  # (n_freqs, n_states)
            if fft_freqs is None:
                fft_freqs = freqs
            if amp.shape[0] == len(fft_freqs):
                spectra.append(amp)
            # trajectories with a different length / sampling rate are
            # skipped from the averaged spectrum (their residual structure
            # still shows up in the residual & scatter plots)

        t_cat = np.concatenate(t_all)
        resid_cat = np.vstack(resid_all)
        dxt_cat = np.vstack(dxt_all)
        dxp_cat = np.vstack(dxp_all)
        fft_amps_avg = np.mean(np.stack(spectra), axis=0) if spectra else None

        result = {
            't': t_cat, 'residuals': {}, 'fft_freqs': fft_freqs,
            'fft_amps': {}, 'dX_true': {}, 'dX_pred': {}, 'stats': {},
        }

        for i in range(resid_cat.shape[1]):
            name = self.feature_names[i] if self.feature_names else f"x{i}"
            result['residuals'][name] = resid_cat[:, i]
            if fft_amps_avg is not None:
                result['fft_amps'][name] = fft_amps_avg[:, i]
            else:
                result['fft_amps'][name] = np.zeros(len(fft_freqs))
            result['dX_true'][name] = dxt_cat[:, i]
            result['dX_pred'][name] = dxp_cat[:, i]

            signal_power = np.var(dxt_cat[:, i])
            noise_power = np.var(resid_cat[:, i])
            r2_dx = float(r2_score(dxt_cat[:, i], dxp_cat[:, i])) \
                if signal_power > 0 else 0.0
            snr_db = 10 * np.log10(signal_power /
                                   noise_power) if noise_power > 0 else 99.0
            r = resid_cat[:, i]
            r_norm = r - r.mean()
            autocorr = float(np.corrcoef(r_norm[:-1], r_norm[1:])[0, 1])

            result['stats'][name] = {
                'r2_dx':    round(r2_dx,   3),
                'snr_db':   round(snr_db,  2),
                'autocorr': round(autocorr, 3),
            }

        return result

    # ------------------------------------------------------------------
    # Block bootstrap helpers — shared logic between split_method="random
    # block" (fit_model) and the ensemble bootstrap below. Chopping into
    # contiguous blocks (instead of resampling individual points) preserves
    # the local time-autocorrelation structure of the trajectory, which a
    # naive point-level bootstrap would destroy.
    # ------------------------------------------------------------------
    def _make_blocks(self, n, n_blocks=20):
        block_size = max(5, n // n_blocks)
        n_blocks_total = n // block_size
        return [np.arange(b * block_size, (b + 1) * block_size)
                for b in range(n_blocks_total)]

    def _make_blocks_multi(self, lengths, n_blocks=20):
        """
        Like _make_blocks, but for a POOLED array stacked from several
        trajectories (same stacking order as _pool_trajectories). Blocks
        are built independently WITHIN each trajectory's own segment, then
        offset into pooled-array coordinates — so a single block can never
        splice together timesteps from two physically unrelated initial
        conditions, which would otherwise defeat the purpose of preserving
        local autocorrelation structure.

        lengths : list of per-trajectory sample counts, in the same order
            they were stacked by _pool_trajectories (e.g. [1000, 850, 1000]).
        """
        blocks = []
        offset = 0
        total_len = sum(lengths)
        for length in lengths:
            # Distribute the ~20-block budget proportionally to each
            # trajectory's share of the total length, at least 1 block
            # per trajectory even if it's short relative to the others.
            n_blocks_i = max(1, round(n_blocks * length / total_len))
            local_blocks = self._make_blocks(length, n_blocks=n_blocks_i)
            blocks.extend([b + offset for b in local_blocks])
            offset += length
        return blocks

    # ------------------------------------------------------------------
    # Block-bootstrap ensemble fit — runs STLSQ n_bootstrap times on
    # resampled-with-replacement BLOCKS of the trajectory/trajectories, to
    # estimate:
    #   - inclusion_pct : how often each candidate term in Theta(X) survives
    #                     thresholding (0 = never, 1 = always)
    #   - coef_mean/std : mean & std of the coefficient, computed ONLY
    #                     over the bootstrap runs where the term was
    #                     non-zero (a term dropped to 0 is a "the model
    #                     chose not to use this" decision, not a small
    #                     coefficient estimate — mixing the two would
    #                     bias the mean toward zero without meaning)
    #
    # CHANGED: multi-trajectory aware, mirrors fit_model()'s pooling logic
    # exactly, so this can run on a model trained from several initial
    # conditions without re-differentiating across a seam where t restarts.
    #
    # Experimental / localhost-only: runs synchronously, no threading.
    # ------------------------------------------------------------------
    def fit_ensemble(self, X, t, poly_degree, threshold, names,
                     lib_type="Polynomial", n_bootstrap=50,
                     random_seed=42, min_inclusion_pct=0.2,
                     progress_callback=None, extra_trajectories=None):
        """
        Parameters
        ----------
        X, t : the PRIMARY trajectory (trajectory #1).
        extra_trajectories : optional list of (X_i, t_i) tuples — any
            ADDITIONAL trajectories that were pooled when the original
            model was trained.

        Returns
        -------
        dict {
            'feature_names': [...],
            'per_state': {
                state_name: {
                    'inclusion_pct': {term: float},
                    'coef_mean':     {term: float or None},
                    'coef_std':      {term: float or None},
                    'n_samples':     {term: int},
                }
            },
            'n_bootstrap': n_bootstrap,
        }
        """
        # --- STEP 0: Build the full trajectory list and pool derivatives ---
        traj_list = [(np.asarray(X, dtype=float), np.asarray(t, dtype=float))]
        for Xi, ti in (extra_trajectories or []):
            traj_list.append((np.asarray(Xi, dtype=float),
                              np.asarray(ti, dtype=float)))
        X_pool, dX_pool, t_pool = self._pool_trajectories(traj_list)
        lengths = [len(ti) for _, ti in traj_list]

        blocks = self._make_blocks_multi(lengths)
        n_blocks = len(blocks)
        rng = np.random.default_rng(random_seed)

        # --- STEP 1: "Probe" fit — NOT a real result, just to get term names ---
        probe_model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.0),
            feature_library=self._build_library(lib_type, poly_degree),
        )
        t_probe = np.arange(len(t_pool), dtype=float)
        probe_model.fit(X_pool, t=t_probe, x_dot=dX_pool, feature_names=names)
        term_names = probe_model.get_feature_names()

        # --- STEP 2: Allocate accumulators ---
        n_states = X_pool.shape[1]
        n_terms = len(term_names)
        inclusion_count = np.zeros((n_states, n_terms))
        coef_records = [[[] for _ in range(n_terms)] for _ in range(n_states)]

        # --- STEP 3: Bootstrap loop ---
        for b in range(n_bootstrap):
            chosen_blocks = rng.choice(n_blocks, size=n_blocks, replace=True)
            idx = np.concatenate([blocks[bi] for bi in chosen_blocks])
            X_b, dX_b = X_pool[idx], dX_pool[idx]
            t_dummy = np.arange(len(idx), dtype=float)

            model_b = ps.SINDy(
                optimizer=ps.STLSQ(threshold=threshold),
                feature_library=self._build_library(lib_type, poly_degree),
                differentiation_method=ps.FiniteDifference()
            )
            try:
                model_b.fit(X_b, t=t_dummy, x_dot=dX_b, feature_names=names)
            except Exception as e:
                print(f"[Ensemble] bootstrap {b} failed: {e}")
                continue

            coefs = model_b.coefficients()
            for s in range(n_states):
                for k in range(n_terms):
                    if coefs[s, k] != 0:
                        inclusion_count[s, k] += 1
                        coef_records[s][k].append(coefs[s, k])

            if progress_callback:
                progress_callback(b + 1, n_bootstrap)

        # --- STEP 4: Aggregate results ---
        result = {'feature_names': term_names,
                  'per_state': {}, 'n_bootstrap': n_bootstrap}

        for s in range(n_states):
            state_name = names[s] if names else f"x{s}"
            incl_pct, coef_mean, coef_std, n_samp = {}, {}, {}, {}

            for k, term in enumerate(term_names):
                incl_pct[term] = float(inclusion_count[s, k] / n_bootstrap)
                vals = coef_records[s][k]
                n_samp[term] = len(vals)

                if incl_pct[term] >= min_inclusion_pct and vals:
                    coef_mean[term] = float(np.mean(vals))
                    coef_std[term] = float(np.std(vals))
                else:
                    coef_mean[term] = None
                    coef_std[term] = None

            result['per_state'][state_name] = {
                'inclusion_pct': incl_pct, 'coef_mean': coef_mean,
                'coef_std': coef_std, 'n_samples': n_samp,
            }
        return result