# AH-FOA: Adaptive Hybrid Falcon Optimization for PV MPPT under Partial Shading

Standalone Python project that proposes and evaluates **AH-FOA**, an
improved variant of the Falcon Optimization Algorithm MPPT introduced
by Alshareef (IEEE Access, 2022), against the standard P&O, PSO, and
vanilla FOA on five partial-shading scenarios.

## Proposed contribution (over Alshareef 2022)

| # | Vanilla FOA limitation | AH-FOA improvement |
|---|---|---|
| 1 | Fixed `AP`, `DP` parameters | Iteration-adaptive `AP(k)`, `DP(k)` |
| 2 | PSO-like search (Eq. 9) stalls in flat regions | Lévy-flight enhanced search step |
| 3 | Persistent steady-state oscillation around GMPP | Hand-off to small-step P&O once swarm spread `std(X) < eps` |
| 4 | Full re-init on irradiance change wastes time | Elitist warm restart: keep `gbest`, reseed only `NP-1` falcons |

## Layout

```
foa_mppt_improved/
  src/
    pv_model.py            # 3-module SDM + bypass diodes (Alshareef Eqs. 1-4)
    converter.py           # quasi-static D -> V_pv operating-point map
    scenarios.py           # 4 PSC patterns (paper) + 1 dynamic case
    simulator.py           # closed-loop sim
    metrics.py             # eta, t_track, ripple, energy yield
    plotting.py
    algorithms/
      pno.py  pso.py  foa.py  ahfoa.py
  scripts/
    run_benchmark.py       # main entry point
    plot_pv_curves.py
  results/                 # CSVs + PNGs land here
```

## Run

```powershell
cd foa_mppt_improved
python -m pip install -r requirements.txt
python -m scripts.plot_pv_curves    # static P-V curves of each case
python -m scripts.run_benchmark     # full benchmark + plots
```

Outputs (`results/`):
- `pv_curves.png` – multi-peak P-V curves of each PSC case.
- `<Case>_traces.png` – power / voltage / duty-cycle traces of all 4 algorithms.
- `<Case>_<Algo>.csv` – raw simulation log per algorithm × case.
- `summary.csv` and `summary_efficiency.png` – consolidated metrics.

## Reference

M.J. Alshareef, *"An Effective Falcon Optimization Algorithm Based MPPT
Under Partial Shaded Photovoltaic Systems"*, **IEEE Access**, vol. 10,
pp. 131345–131360, 2022. doi:10.1109/ACCESS.2022.3226654
