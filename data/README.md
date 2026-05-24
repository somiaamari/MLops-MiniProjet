# 🗄️ Data Directory

## Dataset

**File:** `smart_home_dataset.csv`  
**Size:** ~86 MB  
**Status:** ⚠️ NOT committed to Git (too large — excluded by `.gitignore`)

## How to Get the Dataset

Place the `smart_home_dataset.csv` file in this directory (`data/`) before running the application.

## Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Format: `YYYY-MM-DD HH_MM_SS` |
| `Activity` | string | Current home activity (e.g., sleep, work, leisure) |
| `bed` | int (0/1) | Bed sensor state |
| `tv` | int (0/1) | TV state |
| `oven` | int (0/1) | Oven state |
| `mainDoorLock` | int (0/1) | Main door lock state |
| `bedroomCarp` | int (0/1) | Bedroom carpet sensor |
| `wardrobe` | int (0/1) | Wardrobe sensor |
| `officeLight` | int (0/1) | Office light state |
| `livingLight` | int (0/1) | Living room light state |
| ... | ... | 20+ binary device-state columns |

All device columns are **binary** (0 = OFF, 1 = ON).
