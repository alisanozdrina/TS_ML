# RNO-G ML Training Dataset Builder

This workspace builds HDF5 shards for comparing time-series ML methods with
feature-based methods such as BDTs.

Current working assumption:

- Station 23 real data from 2022 are noise/background: `label = 0`.
- `nc_cr_proxy` simulations are signal: `label = 1`.
- Labels can be refined later without changing the waveform schema.

## Outputs

`build_ml_run_h5.py` writes one HDF5 shard per real run or per simulation NUR file.

Common datasets:

```text
/waveforms              float32 [events, 24, 2048]
/trace_start_time       float64 [events, 24]
/trace_delta_t          float64 [events, 24]
/run                    int32   [events]
/event_number           int32   [events]
/timestamp              float64 [events]
/trigger_type           int8    [events]
/label                  int8    [events]
/source_type            int8    [events]  # 0=data, 1=simulation
/weight                 float32 [events]
/features/snr_avg_deep
/features/coherent_spectral_centroid_pa
/features/n_coincident_pairs_deep
/features/coherent_spectral_peak_frequency_pa
/features/coherent_spectral_entropy_pa
/features/coherent_snr_deep
/features/max_amplitude_avg_pa_over_hpol
/features/impulse_corr_bipolar_wide_avg_pa
/features/impulse_corr_bipolar_avg_pa
/features/surf_corr_zen
/features/rho
/features/phi
/features/z
/features/max_corr
/features/passed_hit_filter
```

Simulation shards also include truth information under `/sim`, for example:

```text
/sim/energy
/sim/shower_energy
/sim/zenith
/sim/azimuth
/sim/flavor
/sim/inelasticity
/sim/triggered
/sim/station_triggered
```

Waveform sample times can be reconstructed with:

```python
times = trace_start_time[event_i, ch_i] + np.arange(2048) * trace_delta_t[event_i, ch_i]
```

## Local Processing

Use the environment where the NuRadioReco RNOG examples run. 

Build real Station 23 data:

```bash
cd /fs/ess/PAS2608/alisa/rno_g/TS_ML
python build_ml_run_h5.py --config build_ml_run_h5.example.yaml
```

Build simulation signal shards:

```bash
python build_ml_run_h5.py --config build_ml_sim.example.yaml
```

Build the train/validation/test manifest after both sets of shards exist:

```bash
python build_training_manifest.py \
  'ml_dataset/data/station23_2022/*.h5' \
  'ml_dataset/sim/nc_cr_proxy/*.h5' \
  --output ml_dataset/training_manifest.csv \
  --relative-to .
```

The manifest is a lightweight index:

```text
file_path,event_index,station,run,event_number,timestamp,trigger_type,label,source_type,split,weight
```

Trigger type codes are stored in both the HDF5 shards and the manifest. The
manifest builder reads `/trigger_type` from each shard:

```text
-1 = unavailable
 1 = LT
 2 = FORCE
 3 = RADIANT
```


## Configs

Real data config:

```text
build_ml_run_h5.example.yaml
```

Important fields:

```yaml
station: 23
year: 2022
input_dir: /fs/ess/PAS2608/rnog/data/full/root
output_dir: /fs/ess/PAS2608/alisa/rno_g/TS_ML/ml_dataset/data/station23_2022
label: 0
source_type: data
reco3d:
  enabled: true
  config_file: /users/PCON0003/anozdrina/rno-g/updatedCRpipeline/deep_cr_search/workflow/configs/reco.yaml
  time_delay_tables: /fs/ess/PAS2608/alisa/rno_g/deepCRsearch/data/multiray_tables
```

Simulation config:

```text
build_ml_sim.example.yaml
```

Important fields:

```yaml
input_files:
  - /fs/ess/PAS2608/alisa/rno_g/deepCRsearch/data/nc_cr_proxy/lgE*/*.nur
output_dir: /fs/ess/PAS2608/alisa/rno_g/TS_ML/ml_dataset/sim/nc_cr_proxy
label: 1
source_type: simulation
use_sim_weights: true
reco3d:
  enabled: true
  config_file: /users/PCON0003/anozdrina/rno-g/updatedCRpipeline/deep_cr_search/workflow/configs/reco.yaml
  time_delay_tables: /fs/ess/PAS2608/alisa/rno_g/deepCRsearch/data/multiray_tables
```

For a smoke test, override the config from the command line:

```bash
python build_ml_run_h5.py --config build_ml_run_h5.example.yaml --run 1000
python build_ml_run_h5.py --config build_ml_sim.example.yaml \
  --input-file /fs/ess/PAS2608/alisa/rno_g/deepCRsearch/data/nc_cr_proxy/lgE19.0/lgE19.0_j0001.nur
```

## Slurm

The script supports Slurm arrays via command-line overrides:

- `--run RUN_NUMBER` for one real data run.
- `--input-file FILE.nur` for one simulation file.
- `--output-dir DIR` if you want to redirect output per campaign.

For real data, submit a direct run range without making a list:

```bash
mkdir -p logs
sbatch --array=0-99%20 \
  --export=ALL,RUN_START=1000,RUN_END=1099 \
  slurm/process_data.sbatch
```

This processes runs `1000` through `1099`. The array index starts at zero, so
task 0 processes `RUN_START`, task 1 processes `RUN_START + 1`, and so on.

If you want an irregular set of runs, make a run list and submit against it:

```bash
mkdir -p slurm/lists logs

printf "1000\n1002\n1017\n" > slurm/lists/station23_runs.txt

sbatch --array=1-$(wc -l < slurm/lists/station23_runs.txt)%20 \
  --export=ALL,RUN_LIST=slurm/lists/station23_runs.txt \
  slurm/process_data.sbatch
```

For simulations, first select files into a list. All files:

```bash
mkdir -p slurm/lists logs
python select_sim_files.py --output slurm/lists/nc_cr_proxy_nur.txt
```

One energy:

```bash
python select_sim_files.py \
  --energy 18.5 \
  --output slurm/lists/nc_cr_proxy_lgE18.5.txt
```

Several energies:

```bash
python select_sim_files.py \
  --energy 18.5,19.0 \
  --output slurm/lists/nc_cr_proxy_lgE18.5_19.0.txt
```

Random subset:

```bash
python select_sim_files.py \
  --energy 18.5,19.0 \
  --n-files 25 \
  --random \
  --seed 23 \
  --output slurm/lists/nc_cr_proxy_random25.txt
```

Then submit the sim array using whichever list you made:

```bash
SIM_LIST=slurm/lists/nc_cr_proxy_random25.txt
sbatch --array=1-$(wc -l < "${SIM_LIST}")%20 \
  --export=ALL,SIM_LIST="${SIM_LIST}" \
  slurm/process_sim.sbatch
```

The `%20` limits concurrency to 20 tasks; adjust it for filesystem and cluster
load.

After both arrays finish, build the manifest:

```bash
sbatch slurm/build_manifest.sbatch
```

Before submitting, edit the `source ...` line in each Slurm file to activate
your NuRadioReco environment.
