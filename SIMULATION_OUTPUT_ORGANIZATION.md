# Simulation Output Organization

## Overview

Each simulation run now creates its own organized subfolder with a unique identifier based on the simulation parameters and timestamp.

## Folder Structure

Instead of all simulations writing to the same folder:
```
output/long_term_uc/monozone_ger/
  ├── data/
  └── figures/
```

Each simulation now gets its own subfolder:
```
output/long_term_uc/monozone_ger/
  ├── 2025_cy1989_1113-1114_20250118_1430/
  │   ├── data/
  │   └── figures/
  ├── 2033_cy2016_0101-1229_20250118_1445/
  │   ├── data/
  │   └── figures/
  └── ...
```

## Folder Naming Convention

The subfolder name format is: `{YEAR}_cy{CLIMATIC_YEAR}_{START_MMDD}-{END_MMDD}_{TIMESTAMP}`

### Components:
- **YEAR**: Target year (2025 or 2033)
- **cy{CLIMATIC_YEAR}**: Climatic year (e.g., cy1989, cy2016)
- **START_MMDD**: Start month-day (e.g., 0101 for Jan 1, 1113 for Nov 13)
- **END_MMDD**: End month-day (calculated from start + duration)
- **TIMESTAMP**: Run timestamp in format YYYYMMDD_HHMM

### Examples:
- `2025_cy1989_1113-1114_20250118_1430` → Year 2025, climatic year 1989, Nov 13-14, run at 14:30 on Jan 18, 2025
- `2033_cy2016_0101-1229_20250118_1445` → Year 2033, climatic year 2016, Jan 1 - Dec 29 (364 days), run at 14:45

## Usage

### Using run_germany.py (Automatic)

Simply edit the configuration at the top of `run_germany.py` and run it:

```python
YEAR = 2033
CLIMATIC_YEAR = 2016
START_MONTH = 1
START_DAY = 1
NUM_DAYS = 364
```

Then run:
```bash
python run_germany.py
```

The script automatically:
1. Generates a unique run ID
2. Prints the output folder path
3. Creates all outputs in that folder

### Using my_toy_ex_germany.py (Optional)

The feature is **enabled by default** in `my_toy_ex_germany.py`. To disable it, set:

```python
USE_SIMULATION_RUN_ID = False  # At line 18
```

When enabled, each run creates its own subfolder just like `run_germany.py`.

## Benefits

1. **No Overwrites**: Each simulation keeps its results separate
2. **Easy Comparison**: Compare results from different configurations side-by-side
3. **Traceability**: Know exactly when and with what parameters each simulation was run
4. **Batch Runs**: Run multiple simulations sequentially without losing previous results

## Technical Details

### Implementation

The feature uses a global variable in `common/long_term_uc_io.py`:
- `set_simulation_run_id(run_id)` - Set the run ID at the start of your script
- `get_simulation_run_id()` - Retrieve the current run ID
- Modified `set_full_lt_uc_output_folder()` to include the run ID in the path

### Backward Compatibility

If no simulation run ID is set (legacy behavior), outputs go directly to `monozone_ger/` as before.

