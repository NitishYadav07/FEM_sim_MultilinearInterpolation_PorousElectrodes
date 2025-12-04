# README: Hierarchical Porous Electrode Simulation Package - FEMMI2025

## Overview

This package simulates ion transport and electrochemical behavior in hierarchical porous electrodes using finite element methods (FEniCS) combined with multilinear interpolation techniques. The code computes concentration profiles, capacitance, energy density, power density, and generates Ragone plots for supercapacitor analysis.

## Installation

### Prerequisites

- **Python 3.8+** with the following packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `sklearn`
  - `meshio`
  - `argparse`
  
- **FEniCS/DOLFINx** (for PDE solving):
  - `dolfinx`
  - `ufl`
  - `basix`
  - `petsc4py`
  - `mpi4py`
  - Optional: `pyvista` and `pyvistaqt` for visualization

- **Bash shell** (Linux/macOS/WSL)

### Installation Steps

1. Install FEniCS/DOLFINx following the official guide: https://fenicsproject.org/download/
2. Install Python dependencies (in a venv):
   ```bash
   pip install numpy pandas matplotlib scipy scikit-learn meshio
   ```
3. Extract the package archive to your desired directory
4. Ensure all shell scripts have execute permissions:
   ```bash
   chmod +x *.sh
   ```

## Directory Structure with brief description

```
.
├── run_HPE_Workflow_GUI_v4_BEST.py     # GUI for workflow execution
├── run_MAIN.sh                          # Master workflow script
├── run_fenics.sh                        # FEniCS simulation driver script
├── run_FitPolynomial_x.sh              # Polynomial fitting driver script
├── run_Interpolate_FittedPoly.sh       # Interpolation script
├── run_Conc3D_to_Conc1D.sh             # (Not included, user-created)
├── runRagonePlot.sh                     # Ragone plot generator script
├── runEnergyPowerRateCharge.sh         # Energy/power calculation driver script
├── run_capacitance_vs_Geom.sh          # Script to plot Capacitance vs geometry
├── run_capacity_vs_Geom.sh             # Script to plot Capacity vs geometry
│
├── PERFECTLYWORKINGwBCs_Mixed_Poisson_2_NPN_v3.py  # FEniCS PDE solver
├── FitPolynomial_x.py                   # Polynomial fitting for training data
├── Interpolate_FittedPoly.py            # Multilinear interpolation engine
├── plot_Conc4Geom.py                    # Concentration plotting (not included)
├── capacityCalculation.py               # Capacity computation
├── capacitanceCalculation.py            # Capacitance computation
├── energyPowerRateChargeCalc.py         # Energy/power metrics
├── ragonePlot.py                        # Ragone plot generation
│
├── FunctionsPorousElectrode.py          # Helper functions (legacy)
├── ParametersPorousElectrode.py         # Parameter definitions (legacy)
├── boundary_condition_class.py          # Boundary condition class
├── functions_NPN.py                     # Mesh conversion utilities
│
├── ElectrodeGeometry/                   # Folder containing the electrode geometry files
│   ├── G1.csv                           # Geometry 1 parameters
│   ├── G2.csv                           # Geometry 2 parameters
│   └── ...                              # (Up to G5.csv here, can be reduced/extended to any number of geometries)
│
├── FeNICsInputFiles/                    # Mesh files for FEniCS
│   ├── L10_w5.msh                       # Mesh for L=10nm, w=5nm
│   └── ...                              # (Various L,w combinations)
│
└── (Output directories created during execution)
    ├── FeNICsOutputFiles/               # FEniCS simulation results
    ├── Main_pore/                       # Main pore interpolated data
    ├── Side_pore/                       # Side pore interpolated data
    ├── capacityResults/                 # Capacity calculation results
    ├── specificCapacitanceResults/      # Capacitance results
    ├── energyPowerResults/              # Energy/power results
    └── ragoneResults/                   # Ragone plot data and images
```

## File Descriptions for GUI

### Main Workflow Scripts

- **`run_HPE_Workflow_GUI_v4_BEST.py`**: Graphical user interface for parameter input and workflow execution
- **`run_MAIN.sh`**: Master script that orchestrates the entire workflow for multiple geometries
- **`run_fenics.sh`**: Executes FEniCS simulations for various pore dimensions and voltages

### Simulation and Analysis files

- **`PERFECTLYWORKINGwBCs_Mixed_Poisson_2_NPN_v3.py`**: Solves coupled Nernst-Planck and Poisson equations using FEniCS
- **`FitPolynomial_x.py`**: Fits 8th-degree polynomials to concentration profiles from FEniCS output
- **`Interpolate_FittedPoly.py`**: Performs 4D multilinear interpolation (L, W, V, c) to predict concentration profiles for arbitrary geometries

### Post-Processing Scripts

- **`capacityCalculation.py`**: Computes specific capacity by integrating concentration profiles
- **`capacitanceCalculation.py`**: Calculates specific capacitance (F/m²) from charge storage
- **`energyPowerRateChargeCalc.py`**: Computes energy density, power density, and charging rate
- **`ragonePlot.py`**: Generates Ragone plots (energy vs. power density)

### Utility Scripts

- **`boundary_condition_class.py`**: Defines boundary condition class for FEniCS
- **`functions_NPN.py`**: Mesh conversion functions (MSH to XDMF)
- **`FunctionsPorousElectrode.py`**: Legacy interpolation functions
- **`ParametersPorousElectrode.py`**: Parameter definitions

## Execution Instructions (Assuming run_fenics.sh has already been run and resulting training data stored in the FeNICsOutputFiles directory)

### Method 1: GUI Execution

1. **Prepare electrode geometry files** in `ElectrodeGeometry/`:
   - Create `G1.csv`, `G2.csv`, etc. with the following format:
     ```
     # Main pore dimensions
     # L (nm), W (nm), X (position)
     100, 10, 0
     # Side pore dimensions (up to 3 side pores)
     20, 5, 10
     30, 5, 50
     25, 5, 80
     ```

2. **Prepare mesh files** in `FeNICsInputFiles/`:
   - Generate `.msh` files using Gmsh for each L,W combination
   - Name format: `L{length}_w{width}.msh`

3. **Run the GUI**:
   ```bash
   python3 run_HPE_Workflow_GUI_v4_BEST.py
   ```

4. **Configure parameters**:
   - Time step (dt): e.g., `1e-06`
   - Number of steps: e.g., `500`
   - Applied voltage (V): e.g., `1.5`
   - Select workflow steps to execute

5. **Execute**: Click "Run Selected Steps" and monitor console output

### Method 2: Command-Line Execution

#### Step 1: Generate Training Data (FeNICs Simulations)

```bash
bash run_fenics.sh <dt> <numsteps> <Vl_low> <Vl_high>
```

Example:
```bash
bash run_fenics.sh 1e-06 500 1 2
```

This runs FEniCS for all combinations of:
- Widths: 5, 10 nm
- Lengths: 10, 20, 40, 50, 60, 80, 100 nm
- Voltages: 1V, 2V

**Output**: `FeNICsOutputFiles/{dt}s/{numsteps}steps/L{L}_W{W}/V{V}/`

#### Step 2: Fit Polynomials to Training Data

```bash
bash run_FitPolynomial_x.sh <dt> <numsteps>
```

Example:
```bash
bash run_FitPolynomial_x.sh 1e-06 500
```

**Output**: Polynomial coefficients in `FeNICsOutputFiles/.../Conc_x_t/fitModel_t_{timestep}.csv`

#### Step 3: Run Main Workflow for Target Geometries

```bash
bash run_MAIN.sh <dt> <numsteps> <voltage>
```

Example:
```bash
bash run_MAIN.sh 1e-06 500 1.5
```

This processes geometries G1 through G5 (configurable via `NGeom` variable in script) and:
1. Interpolates concentration profiles
2. Computes capacity and capacitance
3. Calculates energy/power metrics
4. Generates Ragone plot

**Output directories**:
- `Main_pore/`: Main pore interpolated polynomials
- `Side_pore/`: Side pore interpolated polynomials
- `capacityResults/`
- `specificCapacitanceResults/`
- `energyPowerResults/`
- `ragoneResults/`

### Method 3: Individual Script Execution

```bash
# 1. Interpolate for specific geometry
python3 Interpolate_FittedPoly.py --dt 1e-06 --numsteps 500 --geom 1 --V 1.5

# 2. Calculate capacity
python3 capacityCalculation.py --geom 1

# 3. Calculate capacitance
python3 capacitanceCalculation.py --geom 1 --voltage 1.5

# 4. Calculate energy/power
python3 energyPowerRateChargeCalc.py --geom 1 --voltage 1.5

# 5. Generate Ragone plot
python3 ragonePlot.py --geometries 1 2 3 --voltages 1.0 1.5 2.0
```

## Configuration

### Modifying Number of Geometries

Edit `run_MAIN.sh` and `runRagonePlot.sh`:
```bash
NGeom=5  # Change to your number of geometries
```

### Adjusting Training Grid

Edit `Interpolate_FittedPoly.py`:
```python
L_GRID = np.array([0.0, 10.0, 20.0, 40.0, 50.0, 60.0, 80.0, 100.0])  # nm
W_GRID = np.array([5.0, 10.0])  # nm
V_GRID = np.array([0.0, 1.0, 2.0])  # V
```

### Physical Parameters

Edit `PERFECTLYWORKINGwBCs_Mixed_Poisson_2_NPN_v3.py`:
```python
e = 1.6e-19    # Elementary charge (C)
Z = 1          # Ion valence
D = 1e-8       # Diffusion coefficient (m²/s)
F = 96500      # Faraday constant (C/mol)
T = 300        # Temperature (K)
R = 8.314      # Gas constant (J/mol·K)
```

## Expected Output

After successful execution, you should have:

1. **Capacity plots**: Time evolution of specific capacity (per m²)
2. **Capacitance data**: CSV files with C_spec vs. time
3. **Energy/Power data**: Energy density, power density, and charging rate vs. time
4. **Ragone plot**: PNG image showing energy density vs. power density with geometry annotations

## Troubleshooting

### Common Issues

1. **Missing training data**:
   - Ensure `run_fenics.sh` completed successfully
   - Check that `FeNICsOutputFiles/` contains subdirectories for all L,W,V combinations

2. **File not found errors**:
   - Verify geometry files exist in `ElectrodeGeometry/`
   - Check that polynomial fits were generated in step 2

3. **Interpolation errors**:
   - Ensure target geometry parameters fall within training grid bounds
   - Check that V=0 and L=0 cases are handled (script uses 1e-14 placeholder)

4. **Memory issues**:
   - Reduce number of timesteps or geometries
   - Increase sampling stride in capacity calculations (currently every 10 steps)

5. **MPI/PETSc errors**:
   - Ensure FEniCS is properly installed
   - Try running with single process: `mpirun -np 1 python3 ...`

## Performance Notes

- FEniCS simulations are the most time-consuming step (upto few minutes per configuration)
- Interpolation and post-processing are relatively fast (seconds to minutes)
- Consider running FEniCS simulations on HPC clusters for large parameter sweeps
- The GUI blocks during execution; use terminal for long runs

## Citation

If you use this code in your research, please cite the associated publication in Computer Physics Communications (to be updated soon).

## Support

For issues or questions, please refer to the manuscript documentation or contact the authors.

## License

GPLv3

---

**Last Updated**: 27 November 2025
