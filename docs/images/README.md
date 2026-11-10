# Dashboard screenshots

These are unchanged copies of selected screenshots captured during the measured sessions. The root README uses the two clearest views of the final 32B study; earlier calibration and duplicate views are omitted.

| File | Original capture name | What it shows |
|---|---|---|
| [32B overview](grafana-32b-overview.png) | Screenshot 2026-11-09 011316.png | Calibration followed by the three long cases; activity falls from roughly 32 to at most 16 when the gateway admission limit changes. |
| [32B protected overload](grafana-32b-overload.png) | Screenshot 2026-11-09 011229.png | Final five-minute 4 req/s case with admission limit 16, accepted/rejected rates, GPU memory/utilization, and upstream errors. |
| [7B TP study](grafana-7b-tp-study.png) | Screenshot 2026-11-08 233019.png | Earlier TP study session, including intervals when one or both GPUs were serving. It is a separate model/host session from the 32B study. |

Capture names and displayed times use America/Chicago. Dashboard request rates are rolling estimates; reported request counts and latency targets come from raw client records. GPU utilization is displayed as a fraction, not a percentage.
