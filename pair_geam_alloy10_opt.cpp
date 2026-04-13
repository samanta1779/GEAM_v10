// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Jonathan Zimmerman (Sandia)
------------------------------------------------------------------------- */

#include "pair_geam_alloy10_opt.h"

#include <cmath>
#include <cstring>
#include <climits>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"

#include "neighbor.h"
#include "memory.h"
#include "error.h"
#include "math_special.h"
#include "potential_file_reader.h"
#include "utils.h"



using namespace LAMMPS_NS;
using namespace MathSpecial;

/* ---------------------------------------------------------------------- */

PairGEAMAlloy10::PairGEAMAlloy10(LAMMPS *lmp) : Pair(lmp)
{
  nmax = 0;
  maxshort = 10;
  comm_phase = 0;

  // Model dimensions — set properly in settings(), zeroed here so the
  // destructor is safe if called before settings() runs.
  ngker = 0;
  ngexp = 0;
  ng32 = 0;
  nbasis32 = 0;
  npoly32 = 0;
  nbasis32B = 0;
  nbasis32C = 0;
  nlFlag = 0;
  ngnl = 0;
  ngpair = 0;
  cg32Flag = 0;
  ng32B = 0;
  npoly32B = 0;
  ng32C = 0;
  npoly32C = 0;
  cut = 0.0;
  cutforcesq = 0.0;

  // Pointers
  rho = nullptr;
  rhop = nullptr;
  neighshort = nullptr;
  neigh3f = nullptr;
  psi = nullptr;
  psi2 = nullptr;
  psi4 = nullptr;
  psi2sum = nullptr;
  psi4sum1 = nullptr;

  // comm sizes set to 0 here; actual values are set inside compute()
  // before each forward_comm call. Setting nonzero here would cause
  // LAMMPS to call pack_forward_comm during pair init, before rho is allocated.
  comm_forward = 0;
  comm_reverse = 0;
}

/* ---------------------------------------------------------------------- */

PairGEAMAlloy10::~PairGEAMAlloy10()
{
  memory->destroy(rho);
  memory->destroy(rhop);

  if (nlFlag > 0) {
    memory->destroy(psi);
    memory->destroy(psi2);
    memory->destroy(psi4);
    memory->destroy(psi2sum);
    memory->destroy(psi4sum1);
  }

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(alpha0);
    memory->destroy(beta0);
    memory->destroy(prefac_eam); // destroy line added
    memory->destroy(prefac_NL); // destroy line added
    memory->destroy(beta);
    memory->destroy(c0gker);
    memory->destroy(cgker);
    memory->destroy(cg32);
    memory->destroy(cg32B);
    memory->destroy(cg32C);
    memory->destroy(cpoly32);
    memory->destroy(cpoly32B);
    memory->destroy(cpoly32C);
    memory->destroy(neighshort);

    memory->destroy(s32);
    memory->destroy(s32B);
    memory->destroy(s32C);
    memory->destroy(Gsr32);
    memory->destroy(Gsr32B);
    memory->destroy(Gsr32C);
    memory->destroy(neigh3f);

    memory->destroy(cgrad);
    memory->destroy(c32cA);
    memory->destroy(c32cB);
    memory->destroy(c32cC);
    memory->destroy(cgnl);
    memory->destroy(alpha0nl);
    memory->destroy(beta0nl);
    memory->destroy(betanl);
    memory->destroy(betapair);
    memory->destroy(cgpair);
  }
}

/* max. rel. error <= 1.73e-3 on [-87,88] */
inline double fast_exp(double x)
{
#if 0
  // This seems not accurate enough...
  volatile union {
    float f;
    unsigned int i;
  } cvt;

  if (fabs(x) < 87.0) {

  // exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1
  float t = (float)x * 1.442695041f;
  float fi = floorf (t);
  float f = t - fi;
  int i = (int)fi;
  cvt.f = (0.3371894346f * f + 0.657636276f) * f + 1.00172476f; // compute 2^f
  cvt.i += (i << 23);                                          // scale by 2^i
  return (double)cvt.f;
  } else {
    return exp(x);
  }
#else
  return exp(x);
#endif
}

/* ---------------------------------------------------------------------- */

void PairGEAMAlloy10::compute(int eflag, int vflag)
{
  int i,j,k,ii,jj,kk,n,m,l,g,inum,jnum,nindex;
  int p,q;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double evdwlA, evdwlB, evdwlC;
  double fijvalue, fikvalue, fijkvalue2;
  double rsq,r,tmp,tmp1,tmp2;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double gtmp, stmp, stmpp, rhoip, rhojp; 
  double phitmp, gstmp, rhoip2, rhojp2, tmp3, tmp4;
  double rhopdot, stmpp2, btmp, fx, fy, fz, f1, f2;
  int itype, jtype, ktype, numshort, jnumm1;
  double rij[3], rik[3], rij2, sqrtrij2, rik2, sqrtrik2;
  double fj[3],fk[3];
  double fpair_nl1, fpair_nl2, fpair_nl3, psip_ij, dstmp;
  double etmp_pair, ftmp_pair;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  // grow rho and rhop arrays if necessary
  // need to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(rho);
    memory->destroy(rhop);
    nmax = atom->nmax;
    memory->create(rho, nmax, ngker, "pair:rho");
    memory->create(rhop, nmax, ngker, 3, "pair:rhop");

    if (nlFlag > 0) {
      memory->destroy(psi);
      memory->destroy(psi2);
      memory->destroy(psi4);
      memory->destroy(psi2sum);
      memory->destroy(psi4sum1);
      memory->create(psi, nmax, ngnl, "pair:psi");
      memory->create(psi2, nmax, ngnl, "pair:psi2");
      memory->create(psi4, nmax, ngnl, "pair:psi4");
      memory->create(psi2sum, nmax, ngnl, "pair:psi2sum");
      memory->create(psi4sum1, nmax, ngnl, "pair:psi4sum1");
    }
  }


  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int newton_pair = force->newton_pair;
  int natomtype = atom->ntypes;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // Initialize rho, rhop (contiguous backing storage from memory->create)
  memset(&rho[0][0],     0, (size_t)nall * ngker     * sizeof(double));
  memset(&rhop[0][0][0], 0, (size_t)nall * ngker * 3 * sizeof(double));

  // Initialize psi arrays
  if (nlFlag > 0) {
    memset(&psi[0][0],      0, (size_t)nall * ngnl * sizeof(double));
    memset(&psi2[0][0],     0, (size_t)nall * ngnl * sizeof(double));
    memset(&psi4[0][0],     0, (size_t)nall * ngnl * sizeof(double));
    memset(&psi2sum[0][0],  0, (size_t)nall * ngnl * sizeof(double));
    memset(&psi4sum1[0][0], 0, (size_t)nall * ngnl * sizeof(double));
  }

  // Compute rho
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    etmp_pair = 0.0;
    
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
            
      if (rsq < cutforcesq) {
        if (rsq < 1.0e-20) continue;   // skip overlapping atoms

        r = sqrt(rsq);

        stmp = powint((1-r/cut),4);
        stmpp = 4/(cut*r-rsq);
        
        if (ngpair > 0) {
          for (n=0; n < ngpair; n++) {
            gtmp = fast_exp(-betapair[n]*rsq);
            etmp_pair += cgpair[itype][jtype][n]*gtmp*stmp;
          }
        }

        for (n=0; n < ngker; n++) {
          gtmp = fast_exp(-beta[jtype][n]*rsq);
          phitmp = gtmp*stmp*(2*beta[jtype][n] + stmpp);
          rho[i][n] += gtmp*stmp*prefac_eam[jtype]; // final factor added for prefac
          rhop[i][n][0] -= phitmp*delx*prefac_eam[jtype]; // final factor added for prefac
          rhop[i][n][1] -= phitmp*dely*prefac_eam[jtype]; // final factor added for prefac
          rhop[i][n][2] -= phitmp*delz*prefac_eam[jtype]; // final factor added for prefac
        }

      }
    }

    if (eflag) {
      if (eflag_global) eng_vdwl += etmp_pair/2.;
      if (eflag_atom) eatom[i] += etmp_pair/2.; 
    }
  }

  // communicate rho, rhop
  comm_forward = ngker*4;
  comm_reverse = ngker*4;
  // if (newton_pair) comm->reverse_comm_pair(this);
  comm_phase = 0;
  comm->forward_comm(this);

  // compute Energy due to Em terms 
  if (eflag){
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      itype = type[i];
      tmp = 0.0;
      tmp1 = 0.0;
      for (n=0; n < ngker; n++) {
        rhopdot = rhop[i][n][0]*rhop[i][n][0]
                + rhop[i][n][1]*rhop[i][n][1]
                + rhop[i][n][2]*rhop[i][n][2];
        tmp1 += cgrad[itype][n]* rhopdot;
        tmp2 = 1.0;
        for (m=0; m < ngexp; m++) {
          tmp2 *= rho[i][n];
          tmp += cgker[itype][m][n]*tmp2;
        }
      }
      tmp += c0gker[itype];

      if (eflag_global) eng_vdwl += tmp + tmp1;
      if (eflag_atom) eatom[i] += tmp + tmp1; 
    }
  }


    if (nlFlag > 0) {
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      itype = type[i];
      jlist = firstneigh[i];
      jnum = numneigh[i];
    
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        jtype = type[j];

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
            
        if (rsq < cutforcesq) {
          if (rsq < 1.0e-20) continue;   // skip overlapping atoms

          r = sqrt(rsq);
          stmp = powint((1-r/cut),4);

          for (n=0; n < ngnl; n++){
            gtmp = fast_exp(-betanl[jtype][n]*rsq);
            psi[i][n] += gtmp*stmp;
          }
        }
      }
    }

    // Communicate psi
    comm_forward = ngnl;
    comm_phase = 1;
    comm->forward_comm(this);

    // Compute psi2, psi4
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (n = 0; n < ngnl; n++) {
        tmp = psi[i][n]*psi[i][n];
        psi2[i][n] = tmp;
        psi4[i][n] = tmp*tmp; 
      }
      
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
	    jtype = type[j];

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        
        if (rsq < cutforcesq) {
          if (rsq < 1.0e-20) continue;   // skip overlapping atoms

          r = sqrt(rsq);
          stmp = powint((1-r/cut),4);

          for (n = 0; n < ngnl; n++) {
            tmp = psi[j][n]*psi[j][n];
            psi2[i][n] += tmp*stmp*prefac_NL[jtype];
            psi4[i][n] += tmp*tmp*stmp*prefac_NL[jtype];
          }
        }
      }
      
      tmp = 0.0;
      for (n = 0; n < ngnl; n++) {
        
        psi2[i][n] = sqrt(psi2[i][n]);

        tmp += cgnl[2][n]*pow(psi4[i][n], 0.25) + cgnl[1][n]*pow(psi4[i][n], 0.5) 
             + cgnl[0][n]*psi2[i][n];

        psi2[i][n] = cgnl[0][n]*0.5/psi2[i][n];
        psi4[i][n] = (cgnl[1][n]*0.5/sqrt(psi4[i][n])+cgnl[2][n]*0.25/pow(psi4[i][n],0.75));
      }

      if (eflag_global) eng_vdwl += tmp;
      if (eflag_atom) eatom[i] += tmp;
    }

    // Communicate psi2, psi4
    comm_forward = 2*ngnl;
    comm_phase = 2;
    comm->forward_comm(this);

    // Compute psi2sum, psi4sum
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < cutforcesq) {
          if (rsq < 1.0e-20) continue;   // skip overlapping atoms

          r = sqrt(rsq);
          stmp = powint((1-r/cut),4);

          for (n = 0; n < ngnl; n++) {
            psi2sum[i][n] += psi2[j][n]*stmp;
            psi4sum1[i][n] += psi4[j][n]*stmp;
          }
        }
      }

      itype = type[i];
      for (g = 0; g < ngnl; g++) {
          psi2sum[i][g] = (2*psi[i][g])*(psi2sum[i][g]*prefac_NL[itype]+psi2[i][g]);
          psi2sum[i][g] += (4*powint(psi[i][g],3))*(psi4sum1[i][g]*prefac_NL[itype]+psi4[i][g]);
          // used below: psi,psi2,psi4,psi2sum
      }
    }
  }

  
  // compute Force due to embedding terms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    numshort = 0;

    // Force due to Em term
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      
      if (rsq >= cutforcesq) {
        continue;
      } else {
        if (rsq < 1.0e-20) continue;   // skip overlapping atoms
        neighshort[numshort++] = j;
        if (numshort >= maxshort) {
          maxshort += maxshort/2;
          memory->grow(neighshort,maxshort,"pair:neighshort");

          memory->grow(Gsr32,maxshort*MAX(ng32,1),"pair:Gsr32");
          memory->grow(Gsr32B,maxshort*MAX(ng32B,1),"pair:Gsr32B");
          memory->grow(Gsr32C,maxshort*MAX(ng32C,1),"pair:Gsr32C");
          {
            bigint n3f = (bigint)maxshort * (maxshort-1) / 2 * 10;
            if (n3f < 10) n3f = 10;
            if (n3f > INT_MAX)
              error->one(FLERR,"GEAM neigh3f array size exceeds INT_MAX; reduce cutoff or density");
            memory->grow(neigh3f,(int)n3f,"pair:neigh3f");
          }
        }
      }

      // skip half of the neighbors !!
      if (delx < 0.0) continue;
      if (delx == 0.0 and dely < 0.0) continue;
      if (delx == 0.0 and dely == 0.0 and delz < 0.0) continue;

      fpair = 0.0; // see derivation
      fx = 0.0;
      fy = 0.0;
      fz = 0.0;

      ftmp_pair = 0.0;

      r = sqrt(rsq);
      stmp = powint((1-r/cut),4);
      stmpp = 4/(cut*r-rsq);
      stmpp2 = (stmpp*stmpp/4)*(cut/r-2);

      if (ngpair > 0) {
        for (n = 0; n < ngpair; n++) {
          gtmp = fast_exp(-betapair[n]*rsq);
          gstmp = gtmp*stmp;
          btmp = (2*betapair[n] + stmpp);
          rhoip = gstmp*btmp;
          ftmp_pair += cgpair[itype][jtype][n]*rhoip;
        }
      }

      for (n=0; n < ngker; n++) {

        gtmp = fast_exp(-beta[jtype][n]*rsq);
        gstmp = gtmp*stmp;
        btmp = (2*beta[jtype][n] + stmpp);
        rhoip = gstmp*btmp*prefac_eam[jtype]; // prefactor included (see derivation)
        rhoip2 = -(rhoip*btmp + gstmp*stmpp2);

        gtmp = fast_exp(-beta[itype][n]*rsq);
        gstmp = gtmp*stmp;
        btmp = (2*beta[itype][n] + stmpp);
        rhojp = gstmp*btmp*prefac_eam[itype]; // prefactor included (see derivation)
        rhojp2 = -(rhojp*btmp + gstmp*stmpp2);

        tmp3 = rhop[i][n][0]*delx + rhop[i][n][1]*dely + rhop[i][n][2]*delz;
        tmp4 = rhop[j][n][0]*delx + rhop[j][n][1]*dely + rhop[j][n][2]*delz;
        tmp = cgrad[itype][n]*rhoip2*tmp3 - cgrad[jtype][n]*rhojp2*tmp4;

        f1 = cgrad[itype][n]*rhoip*rhop[i][n][0] - cgrad[jtype][n]*rhojp*rhop[j][n][0];
        f2 = tmp*delx;
        fx += 2*(f1 + f2);
        
        f1 = cgrad[itype][n]*rhoip*rhop[i][n][1] - cgrad[jtype][n]*rhojp*rhop[j][n][1];
        f2 = tmp*dely;
        fy += 2*(f1 + f2);

        f1 = cgrad[itype][n]*rhoip*rhop[i][n][2] - cgrad[jtype][n]*rhojp*rhop[j][n][2];
        f2 = tmp*delz;
        fz += 2*(f1 + f2);

        tmp3 = 1.0;
        tmp4 = 1.0;
        for (m=0; m < ngexp; m++) {
          tmp1 = cgker[itype][m][n]*tmp3*rhoip;
          tmp2 = cgker[jtype][m][n]*tmp4*rhoip;
          fpair += (m+1)*(tmp1+tmp2);
          tmp3 *= rho[i][n];
          tmp4 *= rho[j][n];
        }

      }
      f[i][0] += (delx*(fpair + ftmp_pair) + fx);
      f[i][1] += (dely*(fpair + ftmp_pair) + fy);
      f[i][2] += (delz*(fpair + ftmp_pair) + fz);
      f[j][0] -= (delx*(fpair + ftmp_pair) + fx);
      f[j][1] -= (dely*(fpair + ftmp_pair) + fy);
      f[j][2] -= (delz*(fpair + ftmp_pair) + fz);

      if (evflag) { 
        ev_tally(i,j,nlocal,newton_pair,0.0,0.0,
                             fpair,delx,dely,delz);
        ev_tally(i,j,nlocal,newton_pair,0.0,0.0,
                             ftmp_pair,delx,dely,delz);
        ev_tally_xyz(i,j,nlocal,newton_pair,0.0,0.0,
                          fx,fy,fz,delx,dely,delz);
      }

    }

    // Compute Threebody terms
    evdwlA = 0.0;
    evdwlB = 0.0;
    evdwlC = 0.0;

    for (jj = 0; jj < numshort; jj++) {
      j = neighshort[jj];
      rij[0] = x[j][0] - xtmp;
      rij[1] = x[j][1] - ytmp;
      rij[2] = x[j][2] - ztmp;
      rij2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2];

      for (g = 0; g < ng32; g++) {
        Gsr32[jj*ng32+g] = fast_exp(-s32[g]*rij2);
      }
      for (g = 0; g < ng32B; g++) {
        Gsr32B[jj*ng32B+g] = fast_exp(-s32B[g]*rij2);
      }
      for (g = 0; g < ng32C; g++) {
        Gsr32C[jj*ng32C+g] = fast_exp(-s32C[g]*rij2);
      }
    }
    
    nindex = 0;

    for (jj = 0; jj < numshort; jj++) {
      j = neighshort[jj];
      jtype = type[j];
      rij[0] = x[j][0] - xtmp;
      rij[1] = x[j][1] - ytmp;
      rij[2] = x[j][2] - ztmp;
      rij2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2];
      sqrtrij2 = sqrt(rij2);

      for (kk = jj+1; kk < numshort; kk++) {
        k = neighshort[kk];
        ktype = type[k];
        rik[0] = x[k][0] - xtmp;
        rik[1] = x[k][1] - ytmp;
        rik[2] = x[k][2] - ztmp;
        rik2 = rik[0]*rik[0] + rik[1]*rik[1] + rik[2]*rik[2];
        sqrtrik2 = sqrt(rik2);

        threebody(nindex, jj, kk, rij2, rik2, sqrtrij2, sqrtrik2, rij, rik,
                  itype, jtype, ktype, eflag, fj, fk, evdwlA, 
                  evdwlB, evdwlC, evdwl);
        
        nindex++;
      }
    }

    nindex = 0;

    tmp1 = (c32cB[itype]+evdwlB)*(c32cC[itype]+evdwlC);
    tmp2 = (c32cA[itype]+evdwlA)*(c32cC[itype]+evdwlC);
    tmp3 = (c32cB[itype]+evdwlB)*(c32cA[itype]+evdwlA);

    for (jj = 0; jj < numshort; jj++) {
      j = neighshort[jj];
      rij[0] = x[j][0] - xtmp;
      rij[1] = x[j][1] - ytmp;
      rij[2] = x[j][2] - ztmp;

      for (kk = jj+1; kk < numshort; kk++) {
        k = neighshort[kk];
        rik[0] = x[k][0] - xtmp;
        rik[1] = x[k][1] - ytmp;
        rik[2] = x[k][2] - ztmp;

        fijvalue = neigh3f[10*nindex+1];
        fikvalue = neigh3f[10*nindex+2];
        fijkvalue2 = neigh3f[10*nindex+3];
        fj[0] = (rij[0]*(fijvalue) - rik[0]*fijkvalue2)*tmp1;
        fj[1] = (rij[1]*(fijvalue) - rik[1]*fijkvalue2)*tmp1;
        fj[2] = (rij[2]*(fijvalue) - rik[2]*fijkvalue2)*tmp1;
        fk[0] = (rik[0]*(fikvalue) - rij[0]*fijkvalue2)*tmp1;
        fk[1] = (rik[1]*(fikvalue) - rij[1]*fijkvalue2)*tmp1;
        fk[2] = (rik[2]*(fikvalue) - rij[2]*fijkvalue2)*tmp1;

        if (cg32Flag > 0) {
          fijvalue = neigh3f[10*nindex+4];
          fikvalue = neigh3f[10*nindex+5];
          fijkvalue2 = neigh3f[10*nindex+6];
          fj[0] += (rij[0]*(fijvalue) - rik[0]*fijkvalue2)*tmp2;
          fj[1] += (rij[1]*(fijvalue) - rik[1]*fijkvalue2)*tmp2;
          fj[2] += (rij[2]*(fijvalue) - rik[2]*fijkvalue2)*tmp2;
          fk[0] += (rik[0]*(fikvalue) - rij[0]*fijkvalue2)*tmp2;
          fk[1] += (rik[1]*(fikvalue) - rij[1]*fijkvalue2)*tmp2;
          fk[2] += (rik[2]*(fikvalue) - rij[2]*fijkvalue2)*tmp2;
        }

        if (cg32Flag > 1) {
          fijvalue = neigh3f[10*nindex+7];
          fikvalue = neigh3f[10*nindex+8];
          fijkvalue2 = neigh3f[10*nindex+9];
          fj[0] += (rij[0]*(fijvalue) - rik[0]*fijkvalue2)*tmp3;
          fj[1] += (rij[1]*(fijvalue) - rik[1]*fijkvalue2)*tmp3;
          fj[2] += (rij[2]*(fijvalue) - rik[2]*fijkvalue2)*tmp3;
          fk[0] += (rik[0]*(fikvalue) - rij[0]*fijkvalue2)*tmp3;
          fk[1] += (rik[1]*(fikvalue) - rij[1]*fijkvalue2)*tmp3;
          fk[2] += (rik[2]*(fikvalue) - rij[2]*fijkvalue2)*tmp3;
        }

        f[i][0] -= fj[0] + fk[0];
        f[i][1] -= fj[1] + fk[1];
        f[i][2] -= fj[2] + fk[2];
        f[j][0] += fj[0];
        f[j][1] += fj[1];
        f[j][2] += fj[2];
        f[k][0] += fk[0];
        f[k][1] += fk[1];
        f[k][2] += fk[2];

        evdwl = neigh3f[10*nindex+0];
        if (cg32Flag > 0) {
          if (evflag) ev_tally3(i,j,k,0.0,0.0,fj,fk,rij,rik);
        } else {
          if (evflag) ev_tally3(i,j,k,evdwl,0.0,fj,fk,rij,rik);
        }

        nindex++;
      }
    }

    if (cg32Flag > 0) { // not sure about this, what about evdwlA?
      if (eflag) {
        if (eflag_global) eng_vdwl += (c32cA[itype]+evdwlA)*(c32cB[itype]+evdwlB)*(c32cC[itype]+evdwlC);
        if (eflag_atom) eatom[i] += (c32cA[itype]+evdwlA)*(c32cB[itype]+evdwlB)*(c32cC[itype]+evdwlC);
      }
    }


    // Compute force due to non-linear terms
    if (nlFlag > 0) {

      for (jj = 0; jj < numshort; jj++) {
        j = neighshort[jj];
        jtype = type[j];
        rij[0] = xtmp - x[j][0];
        rij[1] = ytmp - x[j][1];
        rij[2] = ztmp - x[j][2];
        rij2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2];
        sqrtrij2 = sqrt(rij2);

        fpair = 0.0;

        psip_ij = 0.0;
        stmp = powint(1.0 - sqrtrij2/cut,4);
        stmpp = 4.0/(cut*sqrtrij2 - rij2);
        dstmp = stmpp*stmp;

        for (g = 0; g < ngnl; g++) {
          tmp = fast_exp(-betanl[jtype][g]*rij2);
          psip_ij = tmp*stmp*(2*betanl[jtype][g] + stmpp);
          
          fpair += psi2sum[i][g]*psip_ij;
          tmp = psi[j][g]*psi[j][g];
          fpair += psi2[i][g]*tmp*dstmp*prefac_NL[jtype];
          fpair += psi4[i][g]*tmp*tmp*dstmp*prefac_NL[jtype];
        }

        f[i][0] += fpair*rij[0];
        f[i][1] += fpair*rij[1];
        f[i][2] += fpair*rij[2];
        f[j][0] -= fpair*rij[0];
        f[j][1] -= fpair*rij[1];
        f[j][2] -= fpair*rij[2];

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           0.0,0.0,fpair,rij[0],rij[1],rij[2]);

      }

    }

  }

  if (vflag_fdotr) virial_fdotr_compute();
}


/* ----------------------------------------------------------------------
   Threebody Force 
------------------------------------------------------------------------- */
void PairGEAMAlloy10::threebody(int nindex, int jj, int kk, double rij2, double rik2,  
                          double sqrtrij2, double sqrtrik2, double *rij, double *rik, 
                          int itype, int jtype, int ktype, int eflag,
                          double *fj, double *fk, double &evdwlA, double &evdwlB, 
                          double &evdwlC, double &evdwl)
{
  double sumrsq2,tmp,beta2,evalue1,evalue2,evalue2p,dotvalue,cosvalue;
  double fijvalue,fikvalue,fijvalue1,fijvalue2,fikvalue1,fikvalue2;
  double smoothf1;
  double s1, s2, Gm1, Gm2, Gm, smoothfij, smoothfik;
  double dGmij, dGmik, smthfpij, smthfpik, tmp2;

  double evalue1B, fijvalueB, fikvalueB;
  double evalue1C, fijvalueC, fikvalueC;
  double evalue2B, evalue2Bp, evalue2C, evalue2Cp;

  double fijvalue1B,fijvalue2B,fikvalue1B,fikvalue2B;
  double fijvalue1C,fijvalue2C,fikvalue1C,fikvalue2C;
  int i, g, count;

  // initialization
  evdwl = 0.0;

  // compute E32 and f32
  evalue1 = 0.0;
  fijvalue = 0.0;
  fikvalue = 0.0;
  sumrsq2 = rij2 + rik2;
  smoothfij = powint(1.0 - sqrtrij2/cut,4);
  smoothfik = powint(1.0 - sqrtrik2/cut,4);
  smthfpij = 4.0/(cut*sqrtrij2 - rij2)*smoothfij; //-1/r df_ij/dr_ij
  smthfpik = 4.0/(cut*sqrtrik2 - rik2)*smoothfik;

  count = 0;
  for (i = 0; i < ng32; i++) {
    for (g = i; g < ng32; g++) {
      s1 = s32[i];
      s2 = s32[g];

      Gm1 = Gsr32[jj*ng32+i]*Gsr32[kk*ng32+g];
      Gm2 = Gsr32[jj*ng32+g]*Gsr32[kk*ng32+i];
      Gm = Gm1 + Gm2;

      tmp = cg32[itype][jtype][ktype][count]*Gm*smoothfij*smoothfik;
      evalue1 += tmp;

      dGmij = 2*(s1*Gm1 + s2*Gm2);
      dGmik = 2*(s2*Gm1 + s1*Gm2);

      tmp2 = (Gm*smthfpij + dGmij*smoothfij)*smoothfik;
      fijvalue += cg32[itype][jtype][ktype][count]*tmp2;

      tmp2 = (Gm*smthfpik + dGmik*smoothfik)*smoothfij;
      fikvalue += cg32[itype][jtype][ktype][count]*tmp2;

      count++;
    }
  }

  evalue1B = 0.0;
  fijvalueB = 0.0;
  fikvalueB = 0.0;

  if (cg32Flag > 0){
    count = 0;
    for (i = 0; i < ng32B; i++) {
      for (g = i; g < ng32B; g++) {
        s1 = s32B[i];
        s2 = s32B[g];

        Gm1 = Gsr32B[jj*ng32B+i]*Gsr32B[kk*ng32B+g];
        Gm2 = Gsr32B[jj*ng32B+g]*Gsr32B[kk*ng32B+i];
        Gm = Gm1 + Gm2;

        tmp = cg32B[itype][jtype][ktype][count]*Gm*smoothfij*smoothfik;
        evalue1B += tmp;

        dGmij = 2*(s1*Gm1 + s2*Gm2);
        dGmik = 2*(s2*Gm1 + s1*Gm2);

        tmp2 = (Gm*smthfpij + dGmij*smoothfij)*smoothfik;
        fijvalueB += cg32B[itype][jtype][ktype][count]*tmp2;

        tmp2 = (Gm*smthfpik + dGmik*smoothfik)*smoothfij;
        fikvalueB += cg32B[itype][jtype][ktype][count]*tmp2;

        count++;
      }
    }
  }

  evalue1C = 0.0;
  fijvalueC = 0.0;
  fikvalueC = 0.0;

  if (cg32Flag > 1){
    count = 0;
    for (i = 0; i < ng32C; i++) {
      for (g = i; g < ng32C; g++) {
        s1 = s32C[i];
        s2 = s32C[g];

        Gm1 = Gsr32C[jj*ng32C+i]*Gsr32C[kk*ng32C+g];
        Gm2 = Gsr32C[jj*ng32C+g]*Gsr32C[kk*ng32C+i];
        Gm = Gm1 + Gm2;

        tmp = cg32C[itype][jtype][ktype][count]*Gm*smoothfij*smoothfik;
        evalue1C += tmp;

        dGmij = 2*(s1*Gm1 + s2*Gm2);
        dGmik = 2*(s2*Gm1 + s1*Gm2);

        tmp2 = (Gm*smthfpij + dGmij*smoothfij)*smoothfik;
        fijvalueC += cg32C[itype][jtype][ktype][count]*tmp2;

        tmp2 = (Gm*smthfpik + dGmik*smoothfik)*smoothfij;
        fikvalueC += cg32C[itype][jtype][ktype][count]*tmp2;

        count++;
      }
    }
  }

  // compute angular term in E32 and f32
  dotvalue = rij[0]*rik[0] + rij[1]*rik[1] + rij[2]*rik[2];
  cosvalue = dotvalue/(sqrtrij2 * sqrtrik2);

  evalue2 = 1.0;
  evalue2p = 0.0;
  tmp = 1.0;
  for (g = 0; g < npoly32; g++) {
    evalue2p += (g+1)*cpoly32[itype][g]*tmp;
    tmp *= cosvalue;
    evalue2 += cpoly32[itype][g]*tmp;
  }

  evalue2B = 1.0;
  evalue2Bp = 0.0;
  if (cg32Flag > 0){
    tmp = 1.0;
    for (g = 0; g < npoly32B; g++) {
      evalue2Bp += (g+1)*cpoly32B[itype][g]*tmp;
      tmp *= cosvalue;
      evalue2B += cpoly32B[itype][g]*tmp;
    }
  }

  evalue2C = 1.0;
  evalue2Cp = 0.0;
  if (cg32Flag > 1){
    tmp = 1.0;
    for (g = 0; g < npoly32C; g++) {
      evalue2Cp += (g+1)*cpoly32C[itype][g]*tmp;
      tmp *= cosvalue;
      evalue2C += cpoly32C[itype][g]*tmp;
    }
  }

  //if (eflag) evdwl += evalue1*evalue2;
  evdwl += evalue1*evalue2;
  neigh3f[10*nindex+0] = evdwl;

  evdwlA += evalue1*evalue2;
  evdwlB += evalue1B*evalue2B;
  evdwlC += evalue1C*evalue2C;

  fijvalue *= evalue2;
  fikvalue *= evalue2;
  fijvalue2 = fikvalue2 = evalue1 * evalue2p / (sqrtrij2 * sqrtrik2);
  fijvalue += fijvalue2 * dotvalue / rij2;
  fikvalue += fikvalue2 * dotvalue / rik2;

  neigh3f[10*nindex+1] = fijvalue;
  neigh3f[10*nindex+2] = fikvalue;
  neigh3f[10*nindex+3] = fijvalue2;
  

  if (cg32Flag > 0) {
    fijvalueB *= evalue2B;
    fikvalueB *= evalue2B;
    fijvalue2B = fikvalue2B = evalue1B * evalue2Bp / (sqrtrij2 * sqrtrik2);
    fijvalueB += fijvalue2B * dotvalue / rij2;
    fikvalueB += fikvalue2B * dotvalue / rik2;

    neigh3f[10*nindex+4] = fijvalueB;
    neigh3f[10*nindex+5] = fikvalueB;
    neigh3f[10*nindex+6] = fijvalue2B;
  }

  if (cg32Flag > 1) {
    fijvalueC *= evalue2C;
    fikvalueC *= evalue2C;
    fijvalue2C = fikvalue2C = evalue1C * evalue2Cp / (sqrtrij2 * sqrtrik2);
    fijvalueC += fijvalue2C * dotvalue / rij2;
    fikvalueC += fikvalue2C * dotvalue / rik2;

    neigh3f[10*nindex+7] = fijvalueC;
    neigh3f[10*nindex+8] = fikvalueC;
    neigh3f[10*nindex+9] = fijvalue2C;
  }
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairGEAMAlloy10::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;
  // # of arguments depends on dimension of array being allocated
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(alpha0,n+1,"pair:alpha0");
  memory->create(beta0,n+1,"pair:beta0");
  memory->create(beta,n+1,ngker,"pair:beta");
  memory->create(c0gker,n+1,"pair:c0gker");
  memory->create(cgker,n+1,ngexp,ngker,"pair:cgker");
  memory->create(prefac_eam,n+1,"pair:prefac_eam");  // line added for embedding prefactor (1D in # of elements)
  memory->create(prefac_NL,n+1,"pair:prefac_NL");  // line added for NonLinear prefactor (1D in # of elements)

  //memory->create(beta32,ng32,"pair:beta32");
  memory->create(cg32,n+1,n+1,n+1,nbasis32,"pair:cg32");
  memory->create(cg32B,n+1,n+1,n+1,nbasis32B,"pair:cg32B");
  memory->create(cg32C,n+1,n+1,n+1,nbasis32C,"pair:cg32C");
  memory->create(cpoly32,n+1,npoly32,"pair:cpoly32");
  memory->create(cpoly32B,n+1,npoly32B,"pair:cpoly32B");
  memory->create(cpoly32C,n+1,npoly32C,"pair:cpoly32C");
  memory->create(neighshort,maxshort,"pair:neighshort");

  memory->create(s32,ng32,"pair:s32");
  memory->create(s32B,ng32B,"pair:s32B");
  memory->create(s32C,ng32C,"pair:s32C");
  memory->create(Gsr32,maxshort*MAX(ng32,1),"pair:Gsr32");
  memory->create(Gsr32B,maxshort*MAX(ng32B,1),"pair:Gsr32B");
  memory->create(Gsr32C,maxshort*MAX(ng32C,1),"pair:Gsr32C");
  {
    bigint n3f = (bigint)maxshort * (maxshort-1) / 2 * 10;
    if (n3f < 10) n3f = 10;
    if (n3f > INT_MAX)
      error->all(FLERR,"GEAM neigh3f array size exceeds INT_MAX; reduce cutoff or density");
    memory->create(neigh3f,(int)n3f,"pair:neigh3f");
  }
  
  memory->create(cgrad,n+1,ngker,"pair:cgrad");

  memory->create(c32cA,n+1,"pair:c32cA");
  memory->create(c32cB,n+1,"pair:c32cB");
  memory->create(c32cC,n+1,"pair:c32cC");
  
  memory->create(alpha0nl,n+1,"pair:alpha0nl");
  memory->create(beta0nl,n+1,"pair:beta0nl");
  memory->create(betanl,n+1,ngnl,"pair:betanl");
  memory->create(cgnl,3,ngnl,"pair:cgnl");

  memory->create(betapair,ngpair,"pair:betapair");
  memory->create(cgpair,n+1,n+1,ngpair,"pair:cgpair");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairGEAMAlloy10::settings(int narg, char **arg)
{
  if (narg != 13) error->all(FLERR,"Illegal pair_style command");

  cut = utils::numeric(FLERR,arg[0],false,lmp); // cutoff
  cutforcesq = cut * cut;

  // Model parameters
  ngpair = utils::inumeric(FLERR,arg[1],false,lmp); // pair
  ngker = utils::inumeric(FLERR,arg[2],false,lmp); // numG rho
  ngexp = utils::inumeric(FLERR,arg[3],false,lmp); // order of embedding polynomial

  ng32 = utils::inumeric(FLERR,arg[4],false,lmp); // numG 32
  npoly32 = utils::inumeric(FLERR,arg[5],false,lmp); // order of 32 polynomial
  nbasis32 = ng32*(ng32+1)/2;

  nlFlag = utils::inumeric(FLERR,arg[6],false,lmp);
  ngnl = utils::inumeric(FLERR,arg[7],false,lmp); 

  cg32Flag = utils::inumeric(FLERR,arg[8],false,lmp); // number of extra c32

  ng32B = utils::inumeric(FLERR,arg[9],false,lmp); // numG 32B
  npoly32B = utils::inumeric(FLERR,arg[10],false,lmp); // order of 32 polynomialB
  nbasis32B = ng32B*(ng32B+1)/2;

  ng32C = utils::inumeric(FLERR,arg[11],false,lmp); // numG 32C
  npoly32C = utils::inumeric(FLERR,arg[12],false,lmp); // order of 32 polynomialC
  nbasis32C = ng32C*(ng32C+1)/2;

  if (comm->me == 0) {
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairGEAMAlloy10::coeff(int narg, char **arg)
{
  
  if (!allocated) allocate();

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");
  
  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  read_file(arg[2]);

  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      setflag[i][j] = 1;
    }
  }

  for (int g = 0; g < ng32; g++) {
    s32[g] = alpha032*powint(beta032, g);
  }
  for (int g = 0; g < ng32B; g++) {
    s32B[g] = alpha032B*powint(beta032B, g);
  }
  for (int g = 0; g < ng32C; g++) {
    s32C[g] = alpha032C*powint(beta032C, g);
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairGEAMAlloy10::init_style()
{ 
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style GEAM/Alloy10 requires newton pair on");
  
  // need a full neighbor list
  neighbor->add_request(this,NeighConst::REQ_FULL);

  // Set comm sizes to the maximum we will ever need.
  // This must happen here (after settings() has set ngker/ngnl)
  // so that Comm::setup() allocates sufficiently large buffers.
  comm_forward = ngker*4;                         // phase 0: rho + rhop
  if (nlFlag > 0) {
    int ngnl2 = 2*ngnl;                           // phase 2: psi2 + psi4
    if (ngnl2 > comm_forward) comm_forward = ngnl2;
  }
  comm_reverse = ngker*4;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairGEAMAlloy10::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cut;
}

/* ---------------------------------------------------------------------- 
   read the coefficient file
------------------------------------------------------------------------- */

void PairGEAMAlloy10::read_file(char *filename)
{
  int atomid, nelements;
  int natomtype = atom->ntypes;
  int natomtype2 = powint(natomtype+1,2);
  int natomtype3 = powint(natomtype+1,3);

  // read potential file
  if (comm->me == 0) { // if the process is active in processor index 0
    PotentialFileReader reader(lmp, filename, "geam/alloy10", unit_convert_flag);
    
    try {
      ValueTokenizer values = reader.next_values(1);
      nelements =  values.next_int();
 
      if (nelements != atom->ntypes)
        error->one(FLERR,"Incorrect no. of elements in GEAM potential file");
      
      for (int i = 1; i <= atom->ntypes; i++) {

        // extract Element id from file
        values = reader.next_values(1);
        atomid = values.next_int();

        if (atomid < 1 || atomid > atom->ntypes)
          error->one(FLERR,"Incorrect atom id in GEAM potential file");

        values = reader.next_values(2); // added in another value to be read (2 -> 3)

	
        alpha0[atomid] = values.next_double();
        beta0[atomid] = values.next_double();

	values = reader.next_values(1); // line added for reading embedding Z[atomid]
	prefac_eam[atomid] = values.next_double(); // line added for reading embedding Z[atomid]

	values = reader.next_values(1); // line added for reading NL Z[atomid]
	prefac_NL[atomid] = values.next_double(); // line added for reading NL Z[atomid]

  // Read C0       
	values = reader.next_values(1);
        c0gker[atomid] = values.next_double();
 
        for (int m = 0; m < ngexp; m++) {
          reader.next_dvector(&cgker[atomid][m][0], ngker); // vector containing ngker elements for m = m are stored in cgker[atomid][m]
        }
        
  	    for (int n=0; n < ngker; n++) {
	      beta[atomid][n] = alpha0[atomid]*powint(beta0[atomid],n); // calculates nth even-tempered Gaussian
        }                
      }


      // Read three body coefficients
      values = reader.next_values(2);
      alpha032 = values.next_double();
      beta032 = values.next_double(); // alpha0 and beta0 for 32 are scalar values
        

      if (cg32Flag > 0) {
        reader.next_dvector(&c32cA[1], natomtype);
      } else {
        for (int ii=1; ii <= atom->ntypes; ii++){
          c32cA[ii] = 0.0;
        }
      }
      
      for (int ii=1; ii <= atom->ntypes; ii++){
        for (int jj=1; jj <= atom->ntypes; jj++){
          for (int kk=1; kk <= atom->ntypes; kk++){
            reader.next_dvector(&cg32[ii][jj][kk][0], nbasis32);
          }
        }
      }

      int check3body = checkSymm3(cg32, atom->ntypes, nbasis32);
      if (check3body == 1)
	error->one(FLERR,"Cg32 matrix is not symmetic"); // is triggered when C32^alpha,beta,gamma != C32^alpha,gamma,beta

	    for (int ii=1; ii <= atom->ntypes; ii++){
	      reader.next_dvector(&cpoly32[ii][0], npoly32);
	    }

      // Read Coeff for Em2
      for (int ii=1; ii <= atom->ntypes; ii++){
        reader.next_dvector(&cgrad[ii][0], ngker);
      } 
      
      // Coeffs. for non-linear term
      if (nlFlag > 0) {
        for (int i =1; i <= natomtype; i++) {
          values = reader.next_values(2);
          alpha0nl[i] = values.next_double();
          beta0nl[i] = values.next_double();
          for (int n=0; n < ngnl; n++) {
            betanl[i][n] = alpha0nl[i]*powint(beta0nl[i], n);
          }
        }
        
        for (int ii=0; ii < 3; ii++) {
          reader.next_dvector(&cgnl[ii][0], ngnl);
        }
      }
  
      // Coeffs. for pair term
      if (ngpair > 0) {
        values = reader.next_values(2);
        alpha0pair = values.next_double();
        beta0pair = values.next_double();

        for(int n = 0; n < ngpair; n++) {
          betapair[n]=alpha0pair*powint(beta0pair,n);
        }

        for (int ii = 1; ii <= natomtype; ii++) {
          for (int jj = 1; jj <= natomtype; jj++) {
            reader.next_dvector(&cgpair[ii][jj][0], ngpair);
          }
        } 

        check3body = checkSymm2(cgpair, atom->ntypes, ngpair);
        if (check3body == 1)
          error->one(FLERR,"pair is not symmetic");

      }

      if (cg32Flag > 0) {
        values = reader.next_values(2);
        alpha032B = values.next_double();
        beta032B = values.next_double();

        reader.next_dvector(&c32cB[1], natomtype);

        for (int ii=1; ii <= atom->ntypes; ii++){
          for (int jj=1; jj <= atom->ntypes; jj++){
            for (int kk=1; kk <= atom->ntypes; kk++){
              reader.next_dvector(&cg32B[ii][jj][kk][0], nbasis32B);
            }
          }
        }

        check3body = checkSymm3(cg32B, atom->ntypes, nbasis32B);
        if (check3body == 1)
	        error->one(FLERR,"Cg32B matrix is not symmetic"); 

        for (int ii=1; ii <= atom->ntypes; ii++){
          reader.next_dvector(&cpoly32B[ii][0], npoly32B);
        }

      } else {
        for (int ii=1; ii <= atom->ntypes; ii++){
          c32cB[ii] = 1.0;
        }
      }
      
      if (cg32Flag > 1) {
        values = reader.next_values(2);
        alpha032C = values.next_double();
        beta032C = values.next_double();

        reader.next_dvector(&c32cC[1], natomtype);

        for (int ii=1; ii <= atom->ntypes; ii++){
          for (int jj=1; jj <= atom->ntypes; jj++){
            for (int kk=1; kk <= atom->ntypes; kk++){
              reader.next_dvector(&cg32C[ii][jj][kk][0], nbasis32C);
            }
          }
        }

        check3body = checkSymm3(cg32C, atom->ntypes, nbasis32C);
        if (check3body == 1)
	        error->one(FLERR,"cg32C matrix is not symmetic"); 

        for (int ii=1; ii <= atom->ntypes; ii++){
          reader.next_dvector(&cpoly32C[ii][0], npoly32C);
        }

      } else {
        for (int ii=1; ii <= atom->ntypes; ii++){
          c32cC[ii] = 1.0;
        }
      }

    } catch (TokenizerException &e) {
      error->one(FLERR, e.what());
    }
  } // end of comm->me

  // broadcast potential information
  // information read by processor 0 is communicated to all other processors

  MPI_Bcast(&beta[0][0], (natomtype+1)*ngker, MPI_DOUBLE, 0, world);
  MPI_Bcast(&c0gker[0], (natomtype+1), MPI_DOUBLE, 0, world); // order of arguments: pointer to initial memory location, number of units transfered, data type of each unit, source rank (ID of source processor), world
  MPI_Bcast(&cgker[0][0][0], (natomtype+1)*ngexp*ngker, MPI_DOUBLE, 0, world);
  MPI_Bcast(&prefac_eam[0], (natomtype+1), MPI_DOUBLE, 0, world); // line added for embed prefac
  MPI_Bcast(&prefac_NL[0], (natomtype+1), MPI_DOUBLE, 0, world); // line added for NL prefac

  MPI_Bcast(&alpha032, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&beta032, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&alpha032B, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&beta032B, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&alpha032C, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&beta032C, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cg32[0][0][0][0], (natomtype3*nbasis32), MPI_DOUBLE, 0, world);
  MPI_Bcast(&cpoly32[0][0], (natomtype+1)*npoly32, MPI_DOUBLE, 0, world);

  if (cg32Flag > 0) {
    MPI_Bcast(&cg32B[0][0][0][0], (natomtype3*nbasis32B), MPI_DOUBLE, 0, world);
    MPI_Bcast(&cpoly32B[0][0], (natomtype+1)*npoly32B, MPI_DOUBLE, 0, world);
  }

  if (cg32Flag > 1) {
    MPI_Bcast(&cg32C[0][0][0][0], (natomtype3*nbasis32C), MPI_DOUBLE, 0, world);
    MPI_Bcast(&cpoly32C[0][0], (natomtype+1)*npoly32C, MPI_DOUBLE, 0, world);
  }

  MPI_Bcast(&cgrad[0][0], (natomtype+1)*ngker, MPI_DOUBLE, 0, world);

  MPI_Bcast(&c32cA[0], (natomtype+1), MPI_DOUBLE, 0, world);
  MPI_Bcast(&c32cB[0], (natomtype+1), MPI_DOUBLE, 0, world);
  MPI_Bcast(&c32cC[0], (natomtype+1), MPI_DOUBLE, 0, world);

  MPI_Bcast(&betanl[0][0], (natomtype+1)*ngnl, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cgnl[0][0], ngnl*3, MPI_DOUBLE, 0, world);

  if (ngpair > 0) {
    MPI_Bcast(&betapair[0], ngpair, MPI_DOUBLE, 0, world);
    MPI_Bcast(&cgpair[0][0][0], (natomtype+1)*(natomtype+1)*ngpair, MPI_DOUBLE, 0, world);
  }
}

/* ---------------------------------------------------------------------- */

int PairGEAMAlloy10::checkSymm3(double ****c32, int natomtype, int nd4){
  int ii, jj, kk, n;

  for (int ii=1; ii <= natomtype; ii++){
    for (int jj=1; jj <= natomtype; jj++){
      for (int kk=1; kk <= natomtype; kk++){
        for (int n=0; n < nd4; n++) {
          if (c32[ii][jj][kk][n] != c32[ii][kk][jj][n]) {
            return 1;
          }
        }
      }
    }
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

int PairGEAMAlloy10::checkSymm2(double ***c21, int natomtype, int nd4){
  int ii, jj;

  for (int ii=1; ii <= natomtype; ii++){
    for (int jj=1; jj <= natomtype; jj++){
      for (int n=0; n < nd4; n++) {
        if (c21[ii][jj][n] != c21[jj][ii][n]) {
          return 1;
        }
      }
    }
  }
  return 0;
}


/* ---------------------------------------------------------------------- */

double PairGEAMAlloy10::single(int /*i*/, int /*j*/, int itype, int jtype,
                                  double rsq,
                                  double /*factor_coul*/, double factor_lj,
                                  double &fforce)
{
  double phi = 0.0;
  error->all(FLERR, "PairGEAMAlloy10::single is not implemented");

  return phi;
}

/* ---------------------------------------------------------------------- */

int PairGEAMAlloy10::pack_forward_comm(int n, int *list, double *buf,
                               int /*pbc_flag*/, int * /*pbc*/)
{ 
  int i,j,m,ii;
  m = 0;
  if (comm_phase == 0) {
    if (!rho) return 0;    // arrays not yet allocated (called before first compute)
    for (i = 0; i < n; i++) {
      j = list[i];
      for (ii = 0; ii < ngker; ii++){
        buf[m++] = rho[j][ii];
        buf[m++] = rhop[j][ii][0];
        buf[m++] = rhop[j][ii][1];
        buf[m++] = rhop[j][ii][2];
      }
    }
  } else if (comm_phase == 1) {
    for (i = 0; i < n; i++) {
      j = list[i];
      for (ii = 0; ii < ngnl; ii++)
        buf[m++] = psi[j][ii];
    }
  } else if (comm_phase == 2) {
    for (i = 0; i < n; i++) {
      j = list[i];
      for (ii = 0; ii < ngnl; ii++) {
        buf[m++] = psi2[j][ii];
        buf[m++] = psi4[j][ii];
      }
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairGEAMAlloy10::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last,ii;
  m = 0;
  last = first + n;
  if (comm_phase == 0) {
    if (!rho) return;      // arrays not yet allocated (called before first compute)
    for (i = first; i < last; i++)
      for (ii = 0; ii < ngker; ii++) {
        rho[i][ii] = buf[m++];
        rhop[i][ii][0] = buf[m++];
        rhop[i][ii][1] = buf[m++];
        rhop[i][ii][2] = buf[m++];
      }
  } else if (comm_phase == 1) {
    for (i = first; i < last; i++)
      for (ii = 0; ii < ngnl; ii++)
        psi[i][ii] = buf[m++];
  } else if (comm_phase == 2) {
    for (i = first; i < last; i++)
      for (ii = 0; ii < ngnl; ii++) {
        psi2[i][ii] = buf[m++];
        psi4[i][ii] = buf[m++];
      }
  }
}





/* ---------------------------------------------------------------------- */

int PairGEAMAlloy10::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last,ii;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    for (ii = 0; ii < ngker; ii++) {
      buf[m++] = rho[i][ii];
      buf[m++] = rhop[i][ii][0];
      buf[m++] = rhop[i][ii][1];
      buf[m++] = rhop[i][ii][2];  
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairGEAMAlloy10::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m,ii;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (ii = 0; ii < ngker; ii++) {
      rho[j][ii] += buf[m++];
      rhop[j][ii][0] += buf[m++];
      rhop[j][ii][1] += buf[m++];
      rhop[j][ii][2] += buf[m++];
    }
  }
}

/* ---------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairGEAMAlloy10::memory_usage()
{
  double bytes = (double)maxeatom * sizeof(double);
  bytes += (double)maxvatom*6 * sizeof(double);
  bytes += (double)4 * (double)ngker * nmax * sizeof(double);
  bytes += (double)6 * (double)ngnl * nmax * sizeof(double);
  return bytes;
}


/* ----------------------------------------------------------------------
 Radial basis function based potential for multi element system
 Included Terms: 
	1. Embedding energy based term (eam)
	2. General three body interaction with angle (32)
	3. Gradient of embeddng energy term (num_poly_vw)
	4. Concentration based term (emconc) (density approximated by gaussians)
	5. Non-linear term (1 quadratic and 2 quartic terms) (Smmothed when aggregate rho)
	6. Chemistry prefactor added for embedding term
 Pair Style Settings: cut, npolyeam, ngeam, ng32, npoly32...
                      npolyflag, numGsigmaconc, nlFlag, numG_NL
 Neighbor list usage: Full neighbor list 
 Contributing Author: Haoyuan Shi, Bajrang Sharma, Ying Shi Teh, Gopal Iyer, Amit Samanta    
------------------------------------------------------------------------- */
