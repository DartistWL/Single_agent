# HEMS — Execution Representative PPO

## Project Overview

This project implements **Execution Representative Proximal Policy Optimisation (ER-PPO)** for household energy scheduling. It accompanies *Sustainable Energy Scheduling in Smart Homes: A Deep Reinforcement Learning Strategy for Emission Reduction and Economic Optimisation*.

The controller coordinates background demand, a time-shiftable washing machine, an electric vehicle (EV), an energy storage system (ESS), a shared two-zone heat pump and an electric water heater. A common household reward includes electricity expenditure, grid-import emissions, peak demand, device operation and occupant service.

The experiments use 2020 StoreNet demand and PV records from 20 Irish households, measured outdoor temperature and EirGrid carbon intensity. H1–H15 supply training and validation records; H16–H20 are held out for testing. Each episode simulates one household over consecutive days.

## Quick Start

### Requirements

- Python **3.11**.
- PyTorch **2.8.0 with CUDA 12.8**, as pinned in `requirements.txt`.
- NumPy, Pandas, SciPy, Matplotlib, tqdm, tzdata and requests.
- An NVIDIA driver compatible with the installed CUDA-enabled PyTorch build for GPU learning.

The dependency versions are listed in [requirements.txt](requirements.txt).

### Installation

Run the following commands from this project directory:

```bash
conda create -n wl_hems_env python=3.11 -y
conda activate wl_hems_env
python -m pip install -r requirements.txt
```

An existing compatible `wl_hems_env` environment can be activated directly.

### Quick Run

The distributed project contains source code, input data, model checkpoints and exported figures. The evaluation-result directory has been removed. Prepare a new experiment before running training or evaluation; the commands below create new outputs under `evaluation_results/reproduction/` and checkpoints under `model/checkpoints/reproduction/`.

```bash
python -m evaluation list
python -u -m evaluation prepare --episodes 3000 --seeds 11 --output evaluation_results/reproduction
python -u -m evaluation.ER_PPO run --seed 11 --gpu 0 --output evaluation_results/reproduction
python -m evaluation report --algorithm ER_PPO --output evaluation_results/reproduction
```

For a learning method, `run` performs training, validation and test evaluation in sequence. Every command is independent and can be entered on its own line.

## Project Structure

```text
project/
├── data/                       # Input records and data preparation
│   ├── storenet_ireland_2020/   # Demand and PV records for H1–H20
│   └── carbon_2020/             # Measured grid carbon intensity
├── model/
│   ├── paper/                  # ER-PPO and paper comparison controllers
│   ├── ablations/              # IH-PPO component removals
│   ├── common/                 # Networks, PPO updates and execution classes
│   ├── checkpoints/            # Model weights grouped by experiment and seed
│   └── algorithms.json         # Algorithm names and module mapping
├── evaluation/                 # Training/evaluation entry points and reporting
├── analysis/                   # Figure generation and trajectory analysis
├── figure/                     # Exported Fig3–Fig9 in PDF, PNG and SVG
│   ├── supplement/             # Exported supplementary figures
│   └── source/                 # System diagram and workflow source files
├── other/
│   ├── environment/            # Household plant, configuration and weather
│   └── provenance/             # Source identity and archived experiment mapping
├── requirements.txt            # Pinned dependencies
└── README.md
```

New training logs, evaluation traces and summary tables are created during execution. They are not included as a pre-existing results folder.

## Core Features

### 1. Coordinated Household Scheduling

- Fixed background demand, fixed-power appliance appointments and power/thermostat control.
- Battery and thermal states carried across midnight within each scheduling block.
- Joint cost, import-carbon, peak and service objectives.
- Training on 15 households and evaluation on five unseen households, with disjoint dates.

### 2. Execution-Aware Policy Representation

- Physical action masks screen capacity and availability constraints.
- State-dependent execution classes identify commands with the same implemented effect.
- Representative selection with class-size correction preserves the initial legal-command probability mass.
- Six independent actor heads share an observation encoder; a separate critic estimates return.

### 3. Metering and Service Evaluation

- A shared meter accounts for background load, thermal devices, washer, batteries, PV and grid exchanges.
- Battery physical capacity, soft SOC preferences, request repairs and EV service interventions are recorded separately.
- Actual and inventory-adjusted electricity bills are reported together with import emissions and daily peaks.

## Supported Algorithms

### 1. Main Algorithm and Representation Comparisons

| Algorithm | Implementation | Role |
| --- | --- | --- |
| **ER-PPO** | [ER_PPO.py](model/paper/ER_PPO.py) | Representative execution-class policy |
| IH-PPO | [IH_PPO.py](model/paper/IH_PPO.py) | Independent command probabilities |
| IC-PPO | [IC_PPO.py](model/paper/IC_PPO.py) | Independent execution-class probability mass |
| AR-Class PPO | [AR_Class_PPO.py](model/paper/AR_Class_PPO.py) | Conditional execution-class probability mass |

### 2. Reference Learning Methods

| Algorithm | Implementation | Adaptation |
| --- | --- | --- |
| Reference PPO | [Reference_PPO.py](model/paper/Reference_PPO.py) | Reference independent-head PPO |
| CrossQ | [CrossQ.py](model/paper/CrossQ.py) | Continuous outputs mapped to legal device commands |
| DroQ | [DroQ.py](model/paper/DroQ.py) | Dropout Q ensembles with the household action adapter |
| Branching Rainbow | [Branching_Rainbow.py](model/paper/Branching_Rainbow.py) | Device-wise distributional Q branches |

### 3. Conventional Controllers

| Controller | Implementation | Control rule |
| --- | --- | --- |
| SRB | [SRB.py](model/paper/SRB.py) | Service-rule baseline |
| TOU-RBC | [TOU_RBC.py](model/paper/TOU_RBC.py) | Tariff-based scheduling |
| CI-RBC | [CI_RBC.py](model/paper/CI_RBC.py) | Carbon-adjusted tariff rules |
| MPC | [MPC.py](model/paper/MPC.py) | Receding-horizon shooting over eight candidates and up to four half-hours |

### 4. Component Ablations

The component ablations use **IH-PPO** as their parent: `No_SN` removes state normalisation, `No_AN` removes advantage normalisation, `No_Share` removes shared encoding and `No_Mask` removes policy action masking. Their implementations are in [model/ablations/](model/ablations/).

## Usage

### Training Models

After the Quick Start preparation step, use separate terminals for the following independent commands. `--gpu` selects the visible card. The neural-network operations use CUDA; household simulation remains on the CPU.

```bash
python -u -m evaluation.ER_PPO run --seed 11 --gpu 0 --output evaluation_results/reproduction
python -u -m evaluation.IH_PPO run --seed 11 --gpu 1 --output evaluation_results/reproduction
python -u -m evaluation.IC_PPO run --seed 11 --gpu 0 --output evaluation_results/reproduction
python -u -m evaluation.AR_Class_PPO run --seed 11 --gpu 1 --output evaluation_results/reproduction
python -u -m evaluation.Reference_PPO run --seed 11 --gpu 0 --output evaluation_results/reproduction
python -u -m evaluation.CrossQ run --seed 11 --gpu 1 --output evaluation_results/reproduction
python -u -m evaluation.DroQ run --seed 11 --gpu 0 --output evaluation_results/reproduction
python -u -m evaluation.Branching_Rainbow run --seed 11 --gpu 1 --output evaluation_results/reproduction
```

Component ablations can be started independently:

```bash
python -u -m evaluation.No_SN run --seed 11 --gpu 0 --output evaluation_results/reproduction
python -u -m evaluation.No_AN run --seed 11 --gpu 1 --output evaluation_results/reproduction
python -u -m evaluation.No_Share run --seed 11 --gpu 0 --output evaluation_results/reproduction
python -u -m evaluation.No_Mask run --seed 11 --gpu 1 --output evaluation_results/reproduction
```

Different algorithms and seeds can run concurrently. A task lock prevents duplicate writes to the same algorithm, seed and output; there is no configured GPU process-count limit. Repeating a command resumes its committed checkpoint and reuses completed evaluation blocks.

### Evaluating Conventional Controllers

These controllers require no learning and run on the CPU. Each `run` command evaluates validation and test blocks.

```bash
python -u -m evaluation.SRB run --output evaluation_results/reproduction
python -u -m evaluation.TOU_RBC run --output evaluation_results/reproduction
python -u -m evaluation.CI_RBC run --output evaluation_results/reproduction
python -u -m evaluation.MPC run --output evaluation_results/reproduction
```

### Separate Training and Evaluation

Use these stages when training and evaluation need to be managed separately:

```bash
python -u -m evaluation.ER_PPO train --seed 11 --gpu 0 --output evaluation_results/reproduction
python -u -m evaluation.ER_PPO evaluate --seed 11 --gpu 0 --split val --output evaluation_results/reproduction
python -u -m evaluation.ER_PPO evaluate --seed 11 --gpu 0 --split test --output evaluation_results/reproduction
```

### Multiple Training Seeds

Register a separate experiment for the three-seed comparison:

```bash
python -u -m evaluation prepare --episodes 3000 --seeds 11 23 37 --output evaluation_results/reproduction_3seeds
python -u -m evaluation.ER_PPO run --seed 11 --gpu 0 --output evaluation_results/reproduction_3seeds
python -u -m evaluation.ER_PPO run --seed 23 --gpu 1 --output evaluation_results/reproduction_3seeds
python -u -m evaluation.ER_PPO run --seed 37 --gpu 0 --output evaluation_results/reproduction_3seeds
```

Use the corresponding evaluation module for each comparison algorithm and run every registered seed before aggregating its results.

### Summarising Results

After all started runs in the selected output directory have finished:

```bash
python -m evaluation report --output evaluation_results/reproduction
python -m evaluation report --algorithm ER_PPO --output evaluation_results/reproduction_3seeds
```

Reports include aggregate, per-seed and per-household CSV files for validation and test. A single-algorithm report uses an algorithm-specific filename prefix.

### Visualising Results

Existing paper figures are available in [figure/](figure/). Figure generation reads the complete local and repeated experiment records from `evaluation_results/household_multihome_3000/` and `evaluation_results/household_multihome_3seeds/`, including their original manifests, summary tables and traces. These records are not bundled; restore both archived experiment directories before using the plotting commands.

```bash
python -m analysis.plot_figures --supplementary
python -m analysis.plot_figures --language both --supplementary
python -m analysis.Fig3
python -m analysis.Fig7
```

Plots are exported as PDF, PNG and SVG. Chinese plots require a supported Chinese font installed on the machine.

## Configuration Parameters

### Household Configuration

The plant is defined in [other/environment/household.py](other/environment/household.py), with physical transitions in [physics.py](other/environment/physics.py) and shared parameters in [config.json](other/environment/config.json).

| Setting | Value |
| --- | --- |
| Decision interval | 30 minutes; 48 intervals/day |
| Training households | H1–H15 |
| Test households | H16–H20 |
| Training block length | 2–4 consecutive days |
| Evaluation block length | 3 consecutive days |
| ESS / EV capacity | 10 / 60 kWh |
| ESS / EV converter rating | 3.3 / 6.6 kW |
| Battery charge / discharge efficiency | 0.95 / 0.95 |
| Shared heat-pump rating | 3.0 kW |
| Water-heater rating / tank volume | 1.8 kW / 80 L |

### ER-PPO Training Configuration

The policy settings are in [model/common/settings.json](model/common/settings.json).

| Parameter | Value |
| --- | --- |
| Training episodes | 3,000 multi-day blocks |
| Training interactions | 419,952 |
| Validation / test budget | 60 blocks and 180 household-days per split |
| Observation history | 8 |
| Hidden width | 128 |
| Discount / GAE parameter | 0.995 / 0.95 |
| PPO clipping threshold | 0.20 |
| Update epochs / minibatch size | Up to 6 / 128 |
| Initial actor / critic learning rate | 0.0001 / 0.0003 |
| Representative prior | `class_size` |

Different reference methods retain their documented optimiser configurations while sharing the plant interaction budget and evaluation calendar.

## Monitoring Metrics

### Scheduling Performance

- `return_total`: complete household return per household-day in summary tables.
- `cost_eur`: daily metered import expenditure minus export revenue.
- `inventory_aligned_cost_eur`: daily bill with the start/end storage inventory adjustment.
- `carbon_kg`: daily emissions attributed to grid imports using measured intensity.
- `daily_peak_kw`: mean of the daily import maxima.

### Service and Constraints

- `ac_discomfort_hours`: score-weighted AC discomfort duration.
- `hot_water_shortfall_l_c`: drawn water volume multiplied by delivery-temperature deficit.
- `ev_departure_success_rate`: successful departures divided by actual departures.
- `wash_completions`: completed washer cycles per household-day.
- Physical battery excursions, soft SOC-band excursions, request corrections and service interventions are distinguished in the recorded states and action ledger.

### Training Curves

Training CSV files record `block`, `days`, `return_per_day` and cumulative `environment_steps`. Return is normalised by block duration before comparison across training blocks.

## Data Saving

Each prepared experiment creates a `manifest.json` recording the input identity, household split, scheduling calendar, training seeds and calibration. Learning outputs are grouped by seed; deterministic-controller outputs use a separate `controllers/` directory.

Evaluation blocks are identified by both household and start date, for example `H17_2020-11-19.json`. Each block has a completion record and a compressed `.jsonl.gz` trajectory containing observations, requests, executed actions, reward components, meter readings and inventory information. Checkpoints are stored separately under `model/checkpoints/<experiment>/`.

## Notes

1. Run commands from the directory containing `data/`, `model/`, `evaluation/` and `requirements.txt`.
2. Use a new output name, such as `reproduction`, for freshly prepared experiments. Existing archived checkpoints require their matching experiment manifest.
3. Complete preparation before starting independent learning or controller tasks. Calibration uses training records only.
4. Keep registered input files fixed while an experiment runs; use another experiment directory after changing model settings.


## Contact

For questions or suggestions, please contact through:

- Submit an Issue
- Email the project maintainer
