# pair_style geam/alloy10 — Generalized EAM potential framework for LAMMPS

A many-body interatomic potential for multi-element systems implemented as a
LAMMPS pair style.  The model combines an EAM-like embedding energy, explicit
pair interactions, general three-body terms with angular dependence, gradient-of-density
contributions, and non-linear density functionals—all built from even-tempered
Gaussian radial basis functions.

**Contributing authors:** Haoyuan Shi, Bajrang Sharma, Ying Shi Teh, Hong Sun, Liming Zhao, Gopal Iyer, Xiao Han, **Amit Samanta**

**Collaborators:** John Klepeis, Vasily Bulatov, Babak Sadigh, Sebatien Hamel, Vince Lordi, Nicholas Bertin, Ellad Tadmor, Chloe Zeller

---

## Table of contents

1. [Model description](#1-model-description)
2. [LAMMPS syntax](#2-lammps-syntax)
3. [Potential file format](#3-potential-file-format)
4. [Compilation](#4-compilation)
5. [Example input script](#5-example-input-script)
6. [Implementation notes](#6-implementation-notes)

---

## 1. Model description

The total energy of the system is

$$E = E_{\text{pair}} + \sum_i \bigl[ E_{\text{embed}}(i) + E_{\text{grad}}(i) + E_{\text{3body}}(i) + E_{\text{NL}}(i) \bigr]$$

All terms use a common cutoff function $f(r) = (1 - r/r_c)^4$ for $r < r_c$
and $f(r) = 0$ otherwise, where $r_c$ is the global cutoff.

All Gaussian exponents are **even-tempered**: given parameters $\alpha_0$ and
$\beta_0$, the $n$-th exponent is $\beta_n = \alpha_0 \cdot \beta_0^n$.

### 1.1 Pair term (`ngpair > 0`)

$$E_{\text{pair}} = \frac{1}{2} \sum_{i}\sum_{j \neq i} \sum_{n=0}^{N_{\text{pair}}-1} c^{\text{pair}}_{\alpha_i \alpha_j, n} \; e^{-\beta^{\text{pair}}_n r_{ij}^2} \, f(r_{ij})$$

where $\alpha_i$ denotes the element type of atom $i$.  The coefficients satisfy
$c^{\text{pair}}_{\alpha\beta,n} = c^{\text{pair}}_{\beta\alpha,n}$ (pair symmetry).
The exponents are $\beta^{\text{pair}}_n = \alpha_0^{\text{pair}} \cdot (\beta_0^{\text{pair}})^n$.

### 1.2 Embedding energy

A per-atom electron density is defined as a vector of $N_G$ Gaussian channels:

$$\rho_n(i) = \sum_{j \neq i} Z^{\text{eam}}_{\alpha_j} \; e^{-\beta_{\alpha_j,n} \, r_{ij}^2} \, f(r_{ij}), \qquad n = 0, \ldots, N_G - 1$$

where $Z^{\text{eam}}_\alpha$ (`prefac_eam`) is a per-element chemistry prefactor and
$\beta_{\alpha,n} = \alpha_{0,\alpha} \cdot \beta_{0,\alpha}^n$.

The embedding energy is a polynomial in these densities:

$$E_{\text{embed}}(i) = c_0^{(\alpha_i)} + \sum_{n=0}^{N_G-1} \sum_{m=1}^{N_{\text{exp}}} c^{(\alpha_i)}_{m,n} \; [\rho_n(i)]^m$$

### 1.3 Gradient-of-density term

$$E_{\text{grad}}(i) = \sum_{n=0}^{N_G-1} c^{\text{grad}}_{\alpha_i, n} \; |\nabla_i \rho_n(i)|^2$$

where $\nabla_i \rho_n(i)$ is the gradient of the $n$-th density channel with
respect to the position of atom $i$.  This term captures sensitivity to the
spatial arrangement of neighbors beyond what the scalar density encodes.

### 1.4 Three-body term

For each triplet $(i,j,k)$ with $j < k$ in the neighbor list of $i$, the
energy contribution is a product of a radial part and an angular polynomial.

**Radial part** (channel A):

$$V^A_{\text{rad}}(i,j,k) = \sum_{g_1 \le g_2} c^{32}_{\alpha_i \alpha_j \alpha_k, (g_1,g_2)} \; \bigl[G_{g_1}(r_{ij})\,G_{g_2}(r_{ik}) + G_{g_2}(r_{ij})\,G_{g_1}(r_{ik})\bigr] \; f(r_{ij})\,f(r_{ik})$$

where $G_g(r) = e^{-s_g \, r^2}$ with $s_g = \alpha_0^{32} \cdot (\beta_0^{32})^g$.
The radial coefficients satisfy $c^{32}_{\alpha\beta\gamma} = c^{32}_{\alpha\gamma\beta}$
(j-k permutation symmetry).

The basis size is $N_{\text{basis}} = N_{G}^{32}(N_{G}^{32}+1)/2$, corresponding to
the upper-triangular pairs $(g_1, g_2)$.

**Angular part:**

$$P_A(\cos\theta_{jik}) = 1 + \sum_{p=1}^{N_{\text{poly}}} a^{(\alpha_i)}_p \, \cos^p\theta_{jik}$$

where $\cos\theta_{jik} = \hat{r}_{ij} \cdot \hat{r}_{ik}$.

**Simple mode** (`cg32Flag = 0`):

$$E_{\text{3body}}(i) = \sum_{j < k} V^A_{\text{rad}} \cdot P_A$$

**Product mode** (`cg32Flag > 0`):  Channels B and C are defined identically to A
(each with their own Gaussian parameters, coefficients, and angular polynomials).
The three-body energy is a product of three shifted sums:

$$E_{\text{3body}}(i) = \bigl(c_A^{(\alpha_i)} + E_A(i)\bigr) \cdot \bigl(c_B^{(\alpha_i)} + E_B(i)\bigr) \cdot \bigl(c_C^{(\alpha_i)} + E_C(i)\bigr)$$

where $E_X(i) = \sum_{j<k} V^X_{\text{rad}} \cdot P_X$ and $c_X^{(\alpha)}$ are
per-element shift constants.  `cg32Flag = 1` activates channels A+B;
`cg32Flag = 2` activates A+B+C.

### 1.5 Non-linear density term (`nlFlag > 0`)

A second set of smoothed densities is defined:

$$\psi_g(i) = \sum_{j \neq i} e^{-\beta^{\text{NL}}_{\alpha_j, g} \, r_{ij}^2} \, f(r_{ij})$$

These are aggregated non-linearly over the neighborhood:

$$\tilde{\psi}_2(i,g) = \Bigl[\psi_g(i)^2 + \sum_{j \neq i} Z^{\text{NL}}_{\alpha_j} \, \psi_g(j)^2 \, f(r_{ij})\Bigr]^{1/2}$$

$$\tilde{\psi}_4(i,g) = \psi_g(i)^4 + \sum_{j \neq i} Z^{\text{NL}}_{\alpha_j} \, \psi_g(j)^4 \, f(r_{ij})$$

The non-linear energy is:

$$E_{\text{NL}}(i) = \sum_{g=0}^{N_{\text{NL}}-1} \bigl[ c^{(0)}_g \, \tilde{\psi}_2(i,g) + c^{(1)}_g \, \tilde{\psi}_4(i,g)^{1/2} + c^{(2)}_g \, \tilde{\psi}_4(i,g)^{1/4} \bigr]$$

where $Z^{\text{NL}}_\alpha$ (`prefac_NL`) is a per-element prefactor.

---

## 2. LAMMPS syntax

```
pair_style geam/alloy10 cut ngpair ngker ngexp ng32 npoly32 nlFlag ngnl cg32Flag ng32B npoly32B ng32C npoly32C
pair_coeff * * potential_file
```

**Arguments (13 required after `pair_style`):**

| # | Name | Description |
|---|------|-------------|
| 1 | `cut` | Global cutoff distance (Angstrom) |
| 2 | `ngpair` | Number of Gaussian basis functions for the pair term (0 to disable) |
| 3 | `ngker` | Number of Gaussian channels for the EAM density ($N_G$) |
| 4 | `ngexp` | Polynomial order for the embedding energy ($N_{\text{exp}}$) |
| 5 | `ng32` | Number of Gaussians for three-body channel A ($N_G^{32}$) |
| 6 | `npoly32` | Angular polynomial order for channel A |
| 7 | `nlFlag` | Non-linear term: 0 = off, 1 = on |
| 8 | `ngnl` | Number of Gaussians for NL term ($N_{\text{NL}}$; ignored if `nlFlag=0`) |
| 9 | `cg32Flag` | Three-body product mode: 0 = simple, 1 = A×B, 2 = A×B×C |
| 10 | `ng32B` | Gaussians for channel B (ignored if `cg32Flag=0`) |
| 11 | `npoly32B` | Angular polynomial order for channel B |
| 12 | `ng32C` | Gaussians for channel C (ignored if `cg32Flag<2`) |
| 13 | `npoly32C` | Angular polynomial order for channel C |

**Pair coefficients:**

```
pair_coeff * * filename
```

A single potential file covers all element types.  The number of types declared
in the potential file must match the number of atom types in the simulation.

**Requirements:**
- `newton on` (pair style enforces this)
- `atom_style atomic` (or any style providing positions and types)
- `units metal` (energies in eV, distances in Angstrom)

---

## 3. Potential file format

The potential file is read as a whitespace-delimited text file.  Comments are
not supported — every line must contain data.  The file is read by processor 0
and broadcast via MPI.

The sections are read in the following order.  $N_T$ denotes the number of
element types.

```
N_T                                          ← number of element types (must match simulation)

--- Repeat the following block for each element α = 1 .. N_T ---
α                                            ← element ID (integer, 1-indexed)
alpha0[α]   beta0[α]                         ← even-tempered parameters for EAM Gaussians
prefac_eam[α]                                ← chemistry prefactor Z_eam for embedding density
prefac_NL[α]                                 ← chemistry prefactor Z_NL for non-linear term
c0gker[α]                                    ← constant term in embedding polynomial
cgker[α][0][0..ngker-1]                      ← embedding polynomial coefficients, m=0 (one row)
cgker[α][1][0..ngker-1]                      ← m=1
...                                          ← (ngexp rows total, each with ngker values)
--- End per-element block ---

alpha032   beta032                           ← even-tempered params for 3-body channel A
c32cA[1..N_T]                                ← shift constants for product form (read only if cg32Flag > 0)
cg32[1][1][1][0..nbasis32-1]                 ← 3-body radial coefficients: α=1, β=1, γ=1
cg32[1][1][2][0..nbasis32-1]                 ← α=1, β=1, γ=2
...                                          ← (N_T³ rows, iterated as α,β,γ each from 1..N_T)
cpoly32[1][0..npoly32-1]                     ← angular polynomial coefficients, element 1
...                                          ← (N_T rows)

cgrad[1][0..ngker-1]                         ← gradient term coefficients, element 1
...                                          ← (N_T rows)

--- If nlFlag > 0 ---
alpha0nl[1]   beta0nl[1]                     ← NL Gaussian params for element 1
...                                          ← (N_T lines)
cgnl[0][0..ngnl-1]                           ← NL coefficients c^(0)
cgnl[1][0..ngnl-1]                           ← NL coefficients c^(1)
cgnl[2][0..ngnl-1]                           ← NL coefficients c^(2)
--- End NL block ---

--- If ngpair > 0 ---
alpha0pair   beta0pair                       ← even-tempered params for pair Gaussians
cgpair[1][1][0..ngpair-1]                    ← pair coefficients: α=1, β=1
cgpair[1][2][0..ngpair-1]                    ← α=1, β=2
...                                          ← (N_T² rows)
--- End pair block ---

--- If cg32Flag > 0 ---
alpha032B   beta032B                         ← even-tempered params for channel B
c32cB[1..N_T]                                ← shift constants for B
cg32B[...][0..nbasis32B-1]                   ← (N_T³ rows, same iteration order as channel A)
cpoly32B[1..N_T][0..npoly32B-1]              ← (N_T rows)
--- End channel B block ---

--- If cg32Flag > 1 ---
alpha032C   beta032C                         ← even-tempered params for channel C
c32cC[1..N_T]                                ← shift constants for C
cg32C[...][0..nbasis32C-1]                   ← (N_T³ rows)
cpoly32C[1..N_T][0..npoly32C-1]              ← (N_T rows)
--- End channel C block ---
```

**Symmetry requirements (enforced at read time):**
- `cg32[α][β][γ] == cg32[α][γ][β]` for all elements and basis indices (j-k permutation symmetry)
- Same for `cg32B`, `cg32C`
- `cgpair[α][β] == cgpair[β][α]` (pair symmetry)

---

## 4. Compilation

### Prerequisites

- LAMMPS source (stable release 22 Jul 2025 or later recommended)
- C++17 compiler (GCC ≥ 9, Clang ≥ 10, Intel ≥ 2021)
- CMake ≥ 3.16
- MPI library (or serial STUBS)

### Installation

Copy the pair style source files into the LAMMPS `src/` directory:

```bash
cp pair_geam_alloy10_opt.cpp /path/to/lammps/src/
cp pair_geam_alloy10_opt.h   /path/to/lammps/src/
```

LAMMPS will automatically detect the new pair style via the `PairStyle` macro
in the header file.

### Building with CMake

```bash
cd /path/to/lammps
mkdir build && cd build

cmake ../cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MPI=on

make -j$(nproc)
```

For serial builds (no MPI):

```bash
cmake ../cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MPI=off

make -j$(nproc)
```

### Building with the LAMMPS Python interface

To use the pair style from Python (e.g., for the force verification test):

```bash
cmake ../cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=on \
  -DBUILD_MPI=off

make -j$(nproc)
make install

cd /path/to/lammps/python
pip install .
```

### Verification

After building, confirm the pair style is available:

```bash
./lmp -h | grep geam
```

You should see `geam/alloy10` in the list of pair styles.

---

## 5. Example input script

```lammps
units           metal
atom_style      atomic
boundary        p p p

read_data       system.data

mass            1 26.98    # Al
mass            2 63.55    # Cu
mass            3 195.08   # Pt

newton          on

# pair_style geam/alloy10 cut ngpair ngker ngexp ng32 npoly32 nlFlag ngnl cg32Flag ng32B npoly32B ng32C npoly32C
pair_style      geam/alloy10 5.0 2 4 3 3 2 1 2 2 3 2 3 2
pair_coeff      * * AlCuPt.geam

neighbor        1.0 bin
neigh_modify    every 1 delay 0 check yes

velocity        all create 300.0 12345 dist gaussian

fix             1 all nve
thermo          100
timestep        0.001

run             10000
```

---

## 6. Implementation notes

### Neighbor list

The pair style requests a **full** neighbor list (`NeighConst::REQ_FULL`)
and requires `newton on`.  Two-body forces (EAM embedding, gradient, pair) use
a manual half-skip on the full list (based on the sign of the displacement
vector) to avoid double-counting.  Three-body and non-linear forces use the
full list directly.

### Communication pattern

The computation requires multiple forward communication rounds within a single
`compute()` call:

| Phase | `comm_phase` | Data communicated | Size per atom |
|-------|-------------|-------------------|---------------|
| 1 | 0 | `rho[ngker]`, `rhop[ngker][3]` | `4 × ngker` doubles |
| 2 | 1 | `psi[ngnl]` (only if `nlFlag > 0`) | `ngnl` doubles |
| 3 | 2 | `psi2[ngnl]`, `psi4[ngnl]` (only if `nlFlag > 0`) | `2 × ngnl` doubles |

All three phases use the same `pack_forward_comm` / `unpack_forward_comm`
functions, which branch on the `comm_phase` member variable.  The
`comm_forward` member is updated before each call to `comm->forward_comm(this)`.

### Short neighbor list and three-body storage

During the force loop, each atom builds a **short neighbor list** of all
neighbors within the cutoff.  This list is stored in a flat `int` array
(`neighshort`) that grows dynamically via `memory->grow` if needed.

Three-body force prefactors are stored in the `neigh3f` array (10 doubles per
triplet: energy + 3 force components × 3 channels A/B/C).  This array is
written in a first pass over all triplets, then read back in a second pass
where the accumulated energy sums (`evdwlA`, `evdwlB`, `evdwlC`) are available
to form the product-form prefactors `tmp1`, `tmp2`, `tmp3`.

### Even-tempered Gaussian basis

All Gaussian exponents in the model are constructed as geometric sequences:

$$\beta_n = \alpha_0 \cdot \beta_0^n, \qquad n = 0, 1, \ldots, N-1$$

This structure is used for the EAM density (`beta`), three-body radial functions
(`s32`, `s32B`, `s32C`), pair term (`betapair`), and non-linear term (`betanl`).
The exponents are computed once at setup time (in `coeff()` and `read_file()`)
and stored as arrays.

### Cutoff function

The cutoff function $f(r) = (1 - r/r_c)^4$ is computed via `powint` (integer
power) and is identically zero for $r \ge r_c$.  Its derivative factor,
used throughout the force computations, is:

$$\frac{f'(r)}{f(r)} = \frac{-4}{r_c - r} = \frac{4}{r \cdot r_c - r^2}$$

stored as `stmpp` in the code.

### Profiling breakdown

For a representative system (16,000 atoms, 3 element types, all terms active,
`ng32=2`, `ngker=3`, `ngnl=2`, `cg32Flag=2`), the compute time breaks down as:

| Section | % of total |
|---------|-----------|
| Three-body pass 1 (threebody function) | 70% |
| Three-body pass 2 (force application) | 14% |
| EAM / gradient forces | 7% |
| Rho accumulation + communication | 4% |
| NL psi accumulation | 3% |
| Gsr precompute | 3% |

The three-body computation dominates because it scales as O($N_{\text{short}}^2$)
per atom, where $N_{\text{short}}$ is the number of neighbors within the cutoff
(typically 30–60 for metallic systems).

---

## License

This software is distributed under the GNU General Public License as part of
the LAMMPS package.  See the README file in the top-level LAMMPS directory.
