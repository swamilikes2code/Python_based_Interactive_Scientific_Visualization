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
- Upload your own CSV data or choose from built-in pre-set systems
- Configurable library: **Polynomial**, **Fourier**, or **Combined**
- Random train/validation split with per-run metrics (R², RMSE, MAE) in derivative space
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

---

## 📁 Project Structure

```
SINDy/
├── main.py                  # Bokeh + Flask entry point
├── data/                    # Pre-set and user-uploaded CSV files
│   ├── cs_train_data.csv
│   ├── vanderpol_train.csv
│   └── ...
├── engine/
│   └── sindy_model.py       # SINDyEngine class — fit, simulate, diagnostics
└── tabs/
    ├── train_tab.py         # Train & Validate tab layout + callbacks
    ├── test_tab.py          # Test tab layout + callbacks
    └── predict_tab.py       # Predict tab layout + callbacks
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

**requirements.txt** should include:
```
bokeh>=3.0
flask
pysindy
scikit-learn
scipy
numpy
pandas
```

---

## ▶️ Running the App

```bash
bokeh serve --allow-websocket-origin=127.0.0.1:8080 main.py
```

Then open your browser at:
```
http://127.0.0.1:8080
```

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

---

## 🧬 Supported Systems (Pre-set)
The purpose of these two pre-set systems is to show SINDy's ability to retrieve equation from data.
| System | Variables | Dynamics |
|--------|-----------|----------|
| Coupled Spring-Mass | x1, v1, x2, v2 | Linear, oscillatory |
| Van der Pol Oscillator | x, v | Nonlinear limit cycle |

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

## � Design Philosophy

> *"A tool that shows you what it found is useful. A tool that shows you why it found it — and what it might have missed — is a research instrument."*

Three principles guided every design decision:

**Transparency over automation** — The AI Suggester recommends, it does not decide. The Residual Diagnostics shows numbers, it does not classify. The researcher is always the one drawing conclusions.

**Per-run reproducibility** — Every training run is stored with its full diagnostics, plot data, and model instance. Clicking a row in the history table restores everything exactly as it was.

**No silent failures** — Upload errors, simulation divergence, and missing files surface as visible messages, not silent crashes.

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
