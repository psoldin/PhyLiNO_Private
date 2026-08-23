# PhyLiNO

This repository contains the software implementation of the `PhyLiNO` framework described in [PhyLiNO: a forward-folding likelihood-fit framework for neutrino oscillation physics](https://arxiv.org/abs/2502.15253).

PhyLiNO uses a profile-likelihood fit and forward folding to compare parametrized neutrino-oscillation models with experimental datasets. The repository currently contains two experiment modules:

- `LinearRegression`: a self-contained toy example that fits a straight line to an exact Asimov dataset.
- `DoubleChooz`: the experiment-specific forward-folding model. It requires Double Chooz ROOT inputs and a configuration that are not included in this repository.

The instructions below build and run the self-contained linear-regression example.

## Prerequisites

The toy example requires:

- Git, including submodule support;
- CMake 3.14 or newer;
- a C++20 compiler;
- ROOT with Minuit, Minuit2, RooFit, RooFitCore, and RooStats;
- Boost with the filesystem, iostreams, program-options, log, and timer components; and
- internet access during CMake configuration so `nlohmann/json` 3.11.3 can be downloaded.

Eigen is pinned as a Git submodule in `external/eigen` and does not need to be installed separately. MPI is not needed for the toy example and is disabled in the commands below.

## Clone the repository

For a new clone, initialize the Eigen submodule at the same time:

```bash
git clone --recurse-submodules https://github.com/psoldin/PhyLiNO.git
cd PhyLiNO
```

For an existing clone, initialize or update the submodule with:

```bash
git submodule update --init --recursive
```

## Configure and build

From the repository root, configure an out-of-source release build:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI=OFF
```

Then build the executable:

```bash
cmake --build build --parallel
```

The resulting executable is:

```text
build/programs/LLHFit/LLHFit
```

### If ROOT is not discovered automatically

Point `ROOT_DIR` at the directory containing `ROOTConfig.cmake`:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI=OFF \
  -DROOT_DIR=/absolute/path/to/root/cmake
```

Some locally built ROOT installations also require explicit Vdt locations:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI=OFF \
  -DROOT_DIR=/absolute/path/to/root/cmake \
  -DVDT_INCLUDE_DIR=/absolute/path/to/vdt/include \
  -DVDT_LIBRARY=/absolute/path/to/libvdt.so
```

On macOS, the Vdt library normally ends in `.dylib` instead of `.so`.

For an offline build, provide an existing `nlohmann/json` source checkout rather than allowing CMake to download it:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI=OFF \
  -DFETCHCONTENT_SOURCE_DIR_JSON=/absolute/path/to/nlohmann-json
```

## Run the toy fit

Run the executable from the repository root with the supplied configuration:

```bash
./build/programs/LLHFit/LLHFit \
  --config configs/config_linreg.json \
  --silent
```

The example generates 200 exact points from

```text
y = 2*x + 1
```

and fits the slope `a` and intercept `b`. A successful run reports `Fit finished: true`, a likelihood close to zero, and a first fitted parameter close to `2`.

Results are written to `Output.json` in the directory from which the executable is run. The important fields should be approximately:

```json
{
  "converged": true,
  "a": 2.0,
  "b": 1.0,
  "chi2": 0.0
}
```

Small floating-point differences are expected. In the build used to verify these instructions, the fit returned `a = 1.9999999999998104`, `b = 0.9999999999990408`, and `chi2 = 7.88e-20`.

The current `--silent` option suppresses Minuit's detailed output but does not suppress every PhyLiNO status message.

## Run the validation checks

With the standard `build` directory present, run:

```bash
tools/run_validation.sh
```

This rebuilds the executable, verifies that the linear-regression fit recovers its configured truth values, and checks the error messages for missing or unknown experiment names. Use `--no-build` to run the checks without rebuilding:

```bash
tools/run_validation.sh --no-build
```

The Double Chooz regression check needs an external Double Chooz configuration and an untracked baseline output. It is skipped when those files are unavailable; the self-contained toy and configuration checks still run.

## Toy configuration

The supplied [`configs/config_linreg.json`](configs/config_linreg.json) selects the `LinearRegression` module, specifies the Asimov truth and sampling range, and defines the two Minuit parameters. Parameter array order is significant: the linear-regression implementation expects slope `a` first and intercept `b` second.
