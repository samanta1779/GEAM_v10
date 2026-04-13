/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(geam/alloy10,PairGEAMAlloy10);
// clang-format on
#else

#ifndef PAIR_GEAM_ALLOY10OPT_H
#define PAIR_GEAM_ALLOY10OPT_H

#include "pair.h"

namespace LAMMPS_NS {

class PairGEAMAlloy10 : public Pair {
 public:
  PairGEAMAlloy10(class LAMMPS *);
  ~PairGEAMAlloy10() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  double single(int, int, int, int, double, double, double, double &) override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;


  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double memory_usage() override;

 protected:
  int nmax;              // allocated size of per-atom arrays
  int comm_phase;        // selects which data to pack/unpack in forward comm
  int ngker, ngexp;
  int ng32, nbasis32, npoly32;
  int nbasis32B, nbasis32C;
  int nlFlag, ngnl;
  int ngpair;
  int cg32Flag, ng32B, npoly32B, ng32C, npoly32C;
  double cut, cutforcesq;

  double *alpha0, *beta0, *prefac_eam, *prefac_NL, **beta, ***cgker,
      *c0gker;    // prefac chemistry is included here as a 1D array
  double alpha032, beta032, ****cg32, **cpoly32;
  double alpha032B, beta032B, alpha032C, beta032C, ****cg32B, ****cg32C, **cpoly32B, **cpoly32C;
  double *s32, *s32B, *s32C, *Gsr32, *Gsr32B, *Gsr32C;
  double **cgrad;
  double *c32cA, *c32cB, *c32cC;
  double *alpha0nl, *beta0nl, **betanl, **cgnl;
  double alpha0pair, beta0pair, *betapair, ***cgpair;

  int maxshort;       // size of short neighbor list array
  int *neighshort;    // short neighbor list array
  double *neigh3f;

  // per-atom arrays
  double **rho;
  double ***rhop;
  double **psi, **psi2, **psi4;
  double **psi2sum, **psi4sum1;

  void allocate();
  void read_file(char *);
  int checkSymm3(double ****, int, int);
  int checkSymm2(double ***, int, int);
  void threebody(int, int, int, double, double, double, double, double *, double *, int, int, int, int, double *,
                 double *, double &, double &, double &, double &);      
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/
