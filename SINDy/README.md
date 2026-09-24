# 🔬 Lehigh University Impact Fellowship - [STEM VISUALIZATION](https://srrweb.cc.lehigh.edu/app/) - SINDy Module

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Bokeh-3.x-FF7F0E?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/pySINDy-latest-2ECC71?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>An interactive, browser-based tool for identifying governing equations of dynamical systems from data — powered by Sparse Identification of Nonlinear Dynamics (SINDy).</b>
</p>

<p align="center">
  Upload your data. Train a model. Understand what the equations mean. No black boxes.
</p>

---

## ✨ What Is This?

Most equation discovery tools give you a result and walk away. This one stays in the room.

The **SINDy Visualization Module** is a full-stack research tool built for scientists and engineers who want to go beyond just *getting* an equation — they want to *understand* it. From pre-training data analysis to post-training residual diagnostics, every step of the SINDy pipeline is made transparent and interactive.

Whether you are studying coupled oscillators, chaotic attractors, or your own custom dynamical system, this tool gives you the controls and the visibility to do it properly.

---

## 🚀 Features

### 🧠 Train & Validate Tab
- Upload one CSV, upload multiple trajectories of the same system from
  different initial conditions, or choose a built-in pre-set system
- Configurable library: **Polynomial**, **Fourier**, or **Combined**
- Random-sampling, time-based, and random-block train/validation splits
- Multi-trajectory derivatives are computed independently before pooling, so
  time resets between files never create artificial derivative spikes
- Per-file trajectory buttons let users isolate any uploaded initial condition
  without retraining the shared model
- Per-run metrics (R², RMSE, MAE) in derivative space
- Full training history leaderboard — compare runs side by side
- Click any past run to instantly restore its plot and diagnostics
- Delete runs you no longer need

### 🤖 Data Scouting Framework
Before you train, the module analyzes your data and automatically recommends:
- The right **library type** (via FFT-based periodicity detection)
- The optimal **polynomial degree** (via R² comparison at degrees 1, 2, 3)
- A data-driven **sparsity threshold** (via high-frequency noise floor estimation)

No language model. No API calls. Pure signal processing — fast and fully offline.

### 🔬 Residual Diagnostics
After training, three diagnostic plots update automatically:

| Plot | What it reveals |
|------|----------------|
| **Residual vs Time** | Whether the model left behind structured dynamics or just noise |
| **Residual FFT** | The frequency of any missing periodic term in the library |
| **dX True vs dX Predicted** | How well the model captures the derivative across the full dynamic range |

Three quantitative stats accompany every run:
- **R²(dX)** — fraction of derivative variance explained by the model
- **SNR (dB)** — signal-to-noise ratio of the fit in derivative space  
- **Lag-1 Autocorrelation** — whether the residual still contains temporal structure

The diagnostics show you numbers and plots. *You* interpret them. No automated labels, no false confidence.

### 🧪 Test Tab
- Load a trained model from history and evaluate it on unseen test data
- Upload custom test CSV or select from pre-set test files
- Side-by-side comparison: true trajectory vs SINDy simulation
- Per-variable RMSE and R² on state space

### 🔮 Predict Tab
- Run forward simulation from any initial condition using a trained model
- Visualize predicted trajectories interactively

### 🎲 Ensemble Tab
- Run block-bootstrap SINDy fits on any model in training history
- Inspect term inclusion frequency and coefficient mean/standard deviation
- Preserve temporal structure and trajectory boundaries during resampling
- Save and replay multiple ensemble analyses for each trained run
- Report failed bootstrap fits separately; inclusion percentages use only
  successful fits as their denominator

---

## 📁 Project Structure

```
SINDy/
├── main.py                  # Bokeh application and per-session state
├── flask_app.py             # Flask shell that embeds the Bokeh server
├── run.sh                   # Local launcher for both services
├── render.yaml              # Two-service Render deployment blueprint
├── data/                    # Built-in training and test trajectories
├── engine/
│   ├── sindy_model.py       # Fit, simulate, diagnostics, ensemble bootstrap
│   ├── check_datafile.py    # Shared CSV and trajectory-set validation
│   └── suggester.py         # Offline hyperparameter heuristics
├── tabs/
│   ├── train_tab.py         # Train & Validate UI and callbacks
│   ├── test_tab.py          # Held-out trajectory evaluation
│   ├── predict_tab.py       # Forward prediction from a supplied IC
│   └── ensemble_tab.py      # Bootstrap robustness analysis
├── templates/               # Flask page and educational fragments
├── static/                  # CSS and documentation images
└── tests/                   # unittest regression and app smoke tests
```

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/len329lehighedu/STEMVISUALIZATION_SINDy.git
cd STEMVISUALIZATION_SINDy

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## ▶️ Running the App

The local launcher starts both the Bokeh backend and Flask frontend:

```bash
chmod +x run.sh
./run.sh
```

Open:

```
http://127.0.0.1:8080
```

To run the services manually, use two terminals:

```bash
# Terminal 1 — Bokeh backend
bokeh serve --allow-websocket-origin=127.0.0.1:8080 main.py

# Terminal 2 — Flask frontend
python3 flask_app.py
```

### Render deployment

[`render.yaml`](render.yaml) defines two web services:

1. **Bokeh backend** — serves `main.py` and accepts WebSocket connections
   from the Flask service's public hostname.
2. **Flask frontend** — serves the website and embeds the Bokeh application
   through the `BOKEH_URL` environment variable.

Render service names must be globally unique. If Render changes either
hostname, update both `BOKEH_URL` and Bokeh's
`--allow-websocket-origin` value in `render.yaml`, then redeploy.

Training history and uploaded files are held in memory per Bokeh session;
they are intentionally isolated between users and are not persisted after a
session or service restart.

---

## 📊 Data Format

Your CSV must follow this structure:

```
t,       x1,      x2,      ...
0.000,   1.0000,  0.0000,  ...
0.010,   0.9995,  0.0100,  ...
...
```

- First column: **time** (uniformly spaced recommended)
- Remaining columns: **state variables** (any number)
- Header row required — column names become variable names in the equations
- Time must be strictly increasing within each file

### Multiple training trajectories

Multiple uploaded files are treated as trajectories of the **same dynamical
system** measured from different initial conditions. They must have identical
state-column names in identical order. Sample count, duration, time step, and
initial condition may differ between files.

Derivatives and FFTs are computed per trajectory. Training pairs are pooled
only after differentiation, and spectra are aligned to a common frequency grid
before averaging.

Test CSV state columns must exactly match the selected model's training-column
names and order.

---

## 🧬 Supported Systems (Pre-set)

The built-in systems demonstrate equation recovery across different dynamics.

| System | Variables | Dynamics |
|--------|-----------|----------|
| Coupled Spring-Mass | x1, v1, x2, v2 | Linear, oscillatory |
| Van der Pol Oscillator | x, v | Nonlinear limit cycle |
| Nonlinear Pendulum | θ and angular velocity | Trigonometric dynamics |
| Forced Oscillator | System-dependent | Time-dependent/combined library example |

Duffing training trajectories with multiple initial conditions are included in
`data/duffing system/` for custom multi-file experiments.

Custom systems: upload any CSV following the format above.

---

## 🔍 How SINDy Works (In 30 Seconds)

SINDy assumes the system evolves as:

```
dX/dt = f(X)
```

where `f` is a **sparse** combination of candidate functions (the library). Given data `X(t)`, it:

1. Estimates `dX/dt` from the data using smoothed finite differences
2. Builds a library matrix `Θ(X)` — polynomial, Fourier, or combined terms
3. Solves a sparse regression: `dX/dt ≈ Θ(X) · Ξ` where most coefficients in `Ξ` are zero
4. Returns the surviving terms as human-readable equations

The result is an interpretable, parsimonious equation — not a neural network you cannot read.

> **Reference:** Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems.* PNAS.

---

## 🧪 Example Output

After training on the Van der Pol oscillator with a Polynomial (degree 3) library:

```
d(x)/dt =  1.000 v
d(v)/dt = -1.000 x  +  1.000 v  -  1.000 x² v
```

Compared to the true equations:
```
dx/dt = v
dv/dt = μ(1 − x²)v − x        (μ = 1)
```

✅ Exact recovery.

---

## 🧭 Design Philosophy

> *"A tool that shows you what it found is useful. A tool that shows you why it found it — and what it might have missed — is a research instrument."*

Three principles guided every design decision:

**Transparency over automation** — The AI Suggester recommends, it does not decide. The Residual Diagnostics shows numbers, it does not classify. The researcher is always the one drawing conclusions.

**Per-run reproducibility** — Every training run is stored with its full diagnostics, plot data, and model instance. Clicking a row in the history table restores everything exactly as it was.

**No silent failures** — Upload errors, simulation divergence, and missing files surface as visible messages, not silent crashes.

---

## ✅ Automated Tests

The test suite covers multi-trajectory pooling, FFT-grid alignment, residual
segmentation, ensemble failure accounting, prediction initial-condition
compatibility, test-column validation, and construction of all four Bokeh tabs.

```bash
python3 -m unittest discover -s tests -v
```

The suite uses Python's standard-library `unittest`; no additional test
dependency is required.

---

## Read more about this module at: [SINDy_manual](https://docs.google.com/document/d/18w_dNzLmZ-sViatktrZAYshPUXA2W_vM/edit?usp=sharing&ouid=102347802978261295549&rtpof=true&sd=true)


---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

When contributing, please follow these conventions:
- All code comments and docstrings in **English**
- UI text in **English**
- Comment the *why*, not just the *what* — especially for mathematical or signal-processing logic

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Built on top of the excellent [pySINDy](https://github.com/dynamicslab/pysindy) library by the Brunton Lab at the University of Washington.

## Citation

The Ensemble tab implementation is based on the Ensemble-SINDy method proposed in:

> Fasel, U., Kutz, J. N., Brunton, B. W., & Brunton, S. L. (2022). Ensemble-SINDy: 
> Robust sparse model discovery in the low-data, high-noise limit, with active 
> learning and control. *Proceedings of the Royal Society A*, 478(2260), 20210904. 
> https://doi.org/10.1098/rspa.2021.0904
