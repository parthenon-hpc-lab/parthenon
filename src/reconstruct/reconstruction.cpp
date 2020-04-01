//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
//! \file reconstruction.cpp
//  \brief

#include "reconstruction.hpp"
#include "mesh/mesh.hpp"

#include <cmath>      // abs()
#include <cstring>    // strcmp()
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

namespace parthenon {
namespace {
// TODO(felker): replace these hand-rolled linear algebra routines with a real library
constexpr Real lu_tol = 3e-16;
int DoolittleLUPDecompose(Real **a, int n, int *pivot);
void DoolittleLUPSolve(Real **lu, int *pivot, Real *b, int n, Real *x);
} // namespace

// constructor

Reconstruction::Reconstruction(MeshBlock *pmb, ParameterInput *pin) :
    characteristic_projection{false}, uniform{true, true, true},
    // read fourth-order solver switches
    correct_ic{pin->GetOrAddBoolean("time", "correct_ic", false)},
    correct_err{pin->GetOrAddBoolean("time", "correct_err", false)}, pmy_block_{pmb}
{
  // Read and set type of spatial reconstruction
  // --------------------------------
  std::string input_recon = pin->GetOrAddString("mesh", "xorder", "2");

  if (input_recon == "1") {
    xorder = 1;
  } else if (input_recon == "2") {
    xorder = 2;
  } else if (input_recon == "2c") {
    xorder = 2;
    characteristic_projection = true;
  } else if (input_recon == "3") {
    // PPM approximates interfaces with 4th-order accurate stencils, but use xorder=3
    // to denote that the overall scheme is "between 2nd and 4th" order w/o flux terms
    xorder = 3;
  } else if (input_recon == "3c") {
    xorder = 3;
    characteristic_projection = true;
  } else if ((input_recon == "4") || (input_recon == "4c")) {
    xorder = 4;
    if (input_recon == "4c")
      characteristic_projection = true;
    } else if (input_recon == "3") {
      // PPM approximates interfaces with 4th-order accurate stencils, but use xorder=3
      // to denote that the overall scheme is "between 2nd and 4th" order w/o flux terms
      xorder = 3;
    } else if (input_recon == "3c") {
      xorder = 3;
      characteristic_projection = true;
    } else if ((input_recon == "4") || (input_recon == "4c")) {
      // Full 4th-order scheme for hydro or MHD on uniform Cartesian grids
      xorder = 4;
      if (input_recon == "4c")
        characteristic_projection = true;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in Reconstruction constructor" << std::endl
          << "xorder=" << input_recon << " not valid choice for reconstruction"<< std::endl;
      ATHENA_ERROR(msg);
    }
    // Check for incompatible choices with broader solver configuration
    // --------------------------------

    // check for necessary number of ghost zones for PPM w/o fourth-order flux corrections
    if (xorder == 3) {
      int req_nghost = 3;
      if (NGHOST < req_nghost) {
        std::stringstream msg;
        msg << "### FATAL ERROR in Reconstruction constructor" << std::endl
            << "xorder=" << input_recon <<
            " (PPM) reconstruction selected, but nghost=" << NGHOST << std::endl
            << "Reconfigure with --nghost=XXX with XXX > " << req_nghost-1 << std::endl;
        ATHENA_ERROR(msg);
      }
    }

    // perform checks of fourth-order solver configuration restrictions:
    if (xorder == 4) {
      // Uniform, Cartesian mesh with square cells (dx1f=dx2f=dx3f)
      if (pmb->block_size.x1rat != 1.0 || pmb->block_size.x2rat != 1.0 ||
          pmb->block_size.x3rat != 1.0) {
        std::stringstream msg;
        msg << "### FATAL ERROR in Reconstruction constructor" << std::endl
            << "Selected time/xorder=" << input_recon << " flux calculations"
            << " require a uniform (x1rat=x2rat=x3rat=1.0), " << std::endl
            << "Carteisan mesh with square cells. Rerun with uniform cell spacing "
            << std::endl
            << "Current values are:" << std::endl
            << std::scientific
            << std::setprecision(std::numeric_limits<Real>::max_digits10 -1)
            << "x1rat= " << pmb->block_size.x1rat << std::endl
            << "x2rat= " << pmb->block_size.x2rat << std::endl
            << "x3rat= " << pmb->block_size.x3rat << std::endl;
        ATHENA_ERROR(msg);
      }
      Real& dx_i   = pmb->pcoord->dx1f(pmb->cells.x1s(interior));
      Real& dx_j   = pmb->pcoord->dx2f(pmb->cells.x2s(interior));
      Real& dx_k   = pmb->pcoord->dx3f(pmb->cells.x3s(interior));
    // Note, probably want to make the following condition less strict (signal warning
    // for small differences due to floating-point issues) but upgrade to error for
    // large deviations from a square mesh. Currently signals a warning for each
    // MeshBlock with non-square cells.
    if ((pmb->block_size.nx2 > 1 && dx_i != dx_j) ||
        (pmb->block_size.nx3 > 1 && dx_j != dx_k)) {
      // It is possible for small floating-point differences to arise despite equal
      // analytic values for grid spacings in the coordinates.cpp calculation of:
      // Real dx=(block_size.x1max-block_size.x1min)/(ie-is+1);
      // due to the 3x rounding operations in numerator, e.g.
      // float(float(x1max) - float((x1min))
      // if mesh/x1max != mesh/x2max, etc. and/or if an asymmetric MeshBlock
      // decomposition is used
      if (Globals::my_rank == 0) {
        // std::stringstream msg;
        std::cout
            << "### Warning in Reconstruction constructor" << std::endl
            << "Selected time/xorder=" << input_recon << " flux calculations"
            << " require a uniform, Carteisan mesh with" << std::endl
            << "square cells (dx1f=dx2f=dx3f). "
            << "Change mesh limits and/or number of cells for equal spacings\n"
            << "Current values are:" << std::endl
            << std::scientific
            << std::setprecision(std::numeric_limits<Real>::max_digits10 - 1)
            << "dx1f=" << dx_i << std::endl
            << "dx2f=" << dx_j << std::endl
            << "dx3f=" << dx_k << std::endl;
        // ATHENA_ERROR(msg);
      }
    }
    if (pmb->pmy_mesh->multilevel) {
      std::stringstream msg;
      msg << "### FATAL ERROR in Reconstruction constructor" << std::endl
          << "Selected time/xorder=" << input_recon << " flux calculations"
          << " currently does not support SMR/AMR " << std::endl;
      ATHENA_ERROR(msg);
    }

    // check for necessary number of ghost zones for PPM w/ fourth-order flux corrections
    int req_nghost = 4;
    // conversion is added, NGHOST>=6
    if (NGHOST < req_nghost) {
      std::stringstream msg;
      msg << "### FATAL ERROR in Reconstruction constructor" << std::endl
          << "time/xorder=" << input_recon
          << " reconstruction selected, but nghost=" << NGHOST << std::endl
          << "Reconfigure with --nghost=XXX with XXX > " << req_nghost-1 << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  // for all coordinate systems, nonuniform geometric spacing or user-defined
  // MeshGenerator ---> use nonuniform reconstruction weights and limiter terms
  if (pmb->block_size.x1rat != 1.0)
    uniform[X1DIR] = false;
  if (pmb->block_size.x2rat != 1.0)
    uniform[X2DIR] = false;
  if (pmb->block_size.x3rat != 1.0)
    uniform[X3DIR] = false;

  // Uniform mesh with --coord=cartesian or GR: Minkowski, Schwarzschild, Kerr-Schild,
  // GR-User will use the uniform Cartesian limiter and reconstruction weights
  // TODO(c-white): use modified version of curvilinear PPM reconstruction weights and
  // limiter formulations for Schwarzschild, Kerr metrics instead of Cartesian-like wghts

  // Avoid pmb indirection
  const IndexShape cells = pmb->cells;
  // Allocate memory for scratch arrays used in PLM and PPM
  int nc1 = cells.nx1(entire);
  scr01_i_.NewAthenaArray(nc1);
  scr02_i_.NewAthenaArray(nc1);

  scr1_ni_.NewAthenaArray(NWAVE, nc1);
  scr2_ni_.NewAthenaArray(NWAVE, nc1);
  scr3_ni_.NewAthenaArray(NWAVE, nc1);
  scr4_ni_.NewAthenaArray(NWAVE, nc1);

  if ((xorder == 3) || (xorder == 4)) {
    auto &pco = pmb->pcoord;
    scr03_i_.NewAthenaArray(nc1);
    scr04_i_.NewAthenaArray(nc1);
    scr05_i_.NewAthenaArray(nc1);
    scr06_i_.NewAthenaArray(nc1);
    scr07_i_.NewAthenaArray(nc1);
    scr08_i_.NewAthenaArray(nc1);
    scr09_i_.NewAthenaArray(nc1);
    scr10_i_.NewAthenaArray(nc1);
    scr11_i_.NewAthenaArray(nc1);
    scr12_i_.NewAthenaArray(nc1);
    scr13_i_.NewAthenaArray(nc1);
    scr14_i_.NewAthenaArray(nc1);

    scr5_ni_.NewAthenaArray(NWAVE, nc1);
    scr6_ni_.NewAthenaArray(NWAVE, nc1);
    scr7_ni_.NewAthenaArray(NWAVE, nc1);
    scr8_ni_.NewAthenaArray(NWAVE, nc1);

    // Precompute PPM coefficients in x1-direction ---------------------------------------
    c1i.NewAthenaArray(nc1);
    c2i.NewAthenaArray(nc1);
    c3i.NewAthenaArray(nc1);
    c4i.NewAthenaArray(nc1);
    c5i.NewAthenaArray(nc1);
    c6i.NewAthenaArray(nc1);
    hplus_ratio_i.NewAthenaArray(nc1);
    hminus_ratio_i.NewAthenaArray(nc1);

    // Greedily allocate tiny 4x4 matrix + 4x1 vectors (RHS, solution, and permutation
    // indices) in case PPMx1 and/or PPMx2 require them for computing the curvilinear
    // coorddinate reconstruction weights. Same data structures are reused at each spatial
    // index (i or j) and for both PPMx1 and PPMx2 weight calculations:
    constexpr int kNrows = 4;       // = [i-i_L, i+i_R] stencil of reconstruction
    constexpr int kNcols = 4;       // = [0, p-1], p=order of reconstruction
    // system in Mignone equation 21
    Real **beta = new Real*[kNrows];
    for (int i=0; i<kNrows; ++i) {
      beta[i] = new Real[kNcols];
    }

    Real w_sol[kNrows], b_rhs[kNrows];
    int permute[kNrows];
    int m_coord = 2;

    // zero-curvature PPM limiter does not depend on mesh uniformity:
    for (int i=(pmb->cells.x1s(interior))-1; i<=(pmb->cells.x1e(interior))+1; ++i) {
      // h_plus = 3.0;
      // h_minus = 3.0;
      // Ratios are = 2 for Cartesian coords, as in the original PPM limiter's
      // overshoot conditions
      hplus_ratio_i(i) = 2.0;
      hminus_ratio_i(i) = 2.0;
    }
    // 4th order reconstruction weights along Cartesian-like x1 w/ uniform spacing
    if (uniform[X1DIR]) {
#pragma omp simd
      for (int i=cells.x1s(entire); i<=cells.x1e(entire); ++i) {
        // reducing general formula in ppm.cpp corresonds to Mignone eq B.4 weights:
        // (-1/12, 7/12, 7/12, -1/12)
        c1i(i) = 0.5;
        c2i(i) = 0.5;
        c3i(i) = 0.5;
        c4i(i) = 0.5;
        c5i(i) = 1.0/6.0;
        c6i(i) = -1.0/6.0;
      }
    } else { // coeffcients along Cartesian-like x1 with nonuniform mesh spacing
#pragma omp simd
      for (int i=(pmb->cells.x1s(interior))-NGHOST+1; i<=(pmb->cells.x1e(interior))+NGHOST-1; ++i) {
        Real& dx_im1 = pco->dx1f(i-1);
        Real& dx_i   = pco->dx1f(i  );
        Real& dx_ip1 = pco->dx1f(i+1);
        Real qe = dx_i/(dx_im1 + dx_i + dx_ip1);       // Outermost coeff in CW eq 1.7
        c1i(i) = qe*(2.0*dx_im1+dx_i)/(dx_ip1 + dx_i); // First term in CW eq 1.7
        c2i(i) = qe*(2.0*dx_ip1+dx_i)/(dx_im1 + dx_i); // Second term in CW eq 1.7
        if (i > (pmb->cells.x1s(interior))-NGHOST+1) {  // c3-c6 are not computed in first iteration
          Real& dx_im2 = pco->dx1f(i-2);
          Real qa = dx_im2 + dx_im1 + dx_i + dx_ip1;
          Real qb = dx_im1/(dx_im1 + dx_i);
          Real qc = (dx_im2 + dx_im1)/(2.0*dx_im1 + dx_i);
          Real qd = (dx_ip1 + dx_i)/(2.0*dx_i + dx_im1);
          qb = qb + 2.0*dx_i*qb/qa*(qc-qd);
          c3i(i) = 1.0 - qb;
          c4i(i) = qb;
          c5i(i) = dx_i/qa*qd;
          c6i(i) = -dx_im1/qa*qc;
        }
      }
    }

    // Precompute PPM coefficients in x2-direction ---------------------------------------
    if (pmb->block_size.nx2 > 1) {
      int nc2 = cells.nx2(entire);
      c1j.NewAthenaArray(nc2);
      c2j.NewAthenaArray(nc2);
      c3j.NewAthenaArray(nc2);
      c4j.NewAthenaArray(nc2);
      c5j.NewAthenaArray(nc2);
      c6j.NewAthenaArray(nc2);
      hplus_ratio_j.NewAthenaArray(nc2);
      hminus_ratio_j.NewAthenaArray(nc2);

      // zero-curvature PPM limiter does not depend on mesh uniformity:
      for (int j=(pmb->cells.x2s(interior))-1; j<=(pmb->cells.x2e(interior))+1; ++j) {
        // h_plus = 3.0;
        // h_minus = 3.0;
        // Ratios are = 2 for Cartesian coords, as in the original PPM limiter's
        // overshoot conditions
        hplus_ratio_j(j) = 2.0;
        hminus_ratio_j(j) = 2.0;
      }
      // 4th order reconstruction weights along Cartesian-like x2 w/ uniform spacing
      if (uniform[X2DIR]) {
#pragma omp simd
        for (int j=cells.x2s(entire); j<=cells.x2e(entire); ++j) {
          c1j(j) = 0.5;
          c2j(j) = 0.5;
          c3j(j) = 0.5;
          c4j(j) = 0.5;
          c5j(j) = 1.0/6.0;
          c6j(j) = -1.0/6.0;
        }
      } else { // coeffcients along Cartesian-like x2 with nonuniform mesh spacing
#pragma omp simd
        for (int j=(pmb->cells.x2s(interior))-NGHOST+2; j<=(pmb->cells.x2e(interior))+NGHOST-1; ++j) {
          Real& dx_jm1 = pco->dx2f(j-1);
          Real& dx_j   = pco->dx2f(j  );
          Real& dx_jp1 = pco->dx2f(j+1);
          Real qe = dx_j/(dx_jm1 + dx_j + dx_jp1);       // Outermost coeff in CW eq 1.7
          c1j(j) = qe*(2.0*dx_jm1 + dx_j)/(dx_jp1 + dx_j); // First term in CW eq 1.7
          c2j(j) = qe*(2.0*dx_jp1 + dx_j)/(dx_jm1 + dx_j); // Second term in CW eq 1.7

          if (j > (pmb->cells.x2s(interior))-NGHOST+1) {  // c3-c6 are not computed in first iteration
            Real& dx_jm2 = pco->dx2f(j-2);
            Real qa = dx_jm2 + dx_jm1 + dx_j + dx_jp1;
            Real qb = dx_jm1/(dx_jm1 + dx_j);
            Real qc = (dx_jm2 + dx_jm1)/(2.0*dx_jm1 + dx_j);
            Real qd = (dx_jp1 + dx_j)/(2.0*dx_j + dx_jm1);
            qb = qb + 2.0*dx_j*qb/qa*(qc-qd);
            c3j(j) = 1.0 - qb;
            c4j(j) = qb;
            c5j(j) = dx_j/qa*qd;
            c6j(j) = -dx_jm1/qa*qc;
          }
        }
      } // end nonuniform Cartesian-like
    } // end 2D or 3D

    // Precompute PPM coefficients in x3-direction
    if (pmb->block_size.nx3 > 1) {
      int nc3 = cells.nx3(entire);
      c1k.NewAthenaArray(nc3);
      c2k.NewAthenaArray(nc3);
      c3k.NewAthenaArray(nc3);
      c4k.NewAthenaArray(nc3);
      c5k.NewAthenaArray(nc3);
      c6k.NewAthenaArray(nc3);
      hplus_ratio_k.NewAthenaArray(nc3);
      hminus_ratio_k.NewAthenaArray(nc3);

      // reconstruction coeffiencients in x3, Cartesian-like coordinate:
      if (uniform[X3DIR]) { // uniform spacing
#pragma omp simd
        for (int k=cells.x3s(entire); k<=cells.x3e(entire); ++k) {
          c1k(k) = 0.5;
          c2k(k) = 0.5;
          c3k(k) = 0.5;
          c4k(k) = 0.5;
          c5k(k) = 1.0/6.0;
          c6k(k) = -1.0/6.0;
        }

      } else { // nonuniform spacing
#pragma omp simd
        for (int k=(pmb->cells.x3s(interior))-NGHOST+2; k<=(pmb->cells.x3e(interior))+NGHOST-1; ++k) {
          Real& dx_km1 = pco->dx3f(k-1);
          Real& dx_k   = pco->dx3f(k  );
          Real& dx_kp1 = pco->dx3f(k+1);
          Real qe = dx_k/(dx_km1 + dx_k + dx_kp1);       // Outermost coeff in CW eq 1.7
          c1k(k) = qe*(2.0*dx_km1+dx_k)/(dx_kp1 + dx_k); // First term in CW eq 1.7
          c2k(k) = qe*(2.0*dx_kp1+dx_k)/(dx_km1 + dx_k); // Second term in CW eq 1.7

          if (k > (pmb->cells.x3s(interior))-NGHOST+1) {  // c3-c6 are not computed in first iteration
            Real& dx_km2 = pco->dx3f(k-2);
            Real qa = dx_km2 + dx_km1 + dx_k + dx_kp1;
            Real qb = dx_km1/(dx_km1 + dx_k);
            Real qc = (dx_km2 + dx_km1)/(2.0*dx_km1 + dx_k);
            Real qd = (dx_kp1 + dx_k)/(2.0*dx_k + dx_km1);
            qb = qb + 2.0*dx_k*qb/qa*(qc-qd);
            c3k(k) = 1.0 - qb;
            c4k(k) = qb;
            c5k(k) = dx_k/qa*qd;
            c6k(k) = -dx_km1/qa*qc;
          }
        }
        // Compute geometric factors for x3 limiter (Mignone eq 48)
        // (no curvilinear corrections in x3)
        for (int k=(pmb->cells.x3s(interior))-1; k<=(pmb->cells.x3e(interior))+1; ++k) {
          // h_plus = 3.0;
          // h_minus = 3.0;
          // Ratios are both = 2 for Cartesian and all curviliniear coords
          hplus_ratio_k(k) = 2.0;
          hminus_ratio_k(k) = 2.0;
        }
      }
    }
    for (int i=0; i<kNrows; ++i) {
      delete[] beta[i];
    }
    delete[] beta;
  } // end "if PPM or full 4th order spatial integrator"
}


namespace {

//----------------------------------------------------------------------------------------
// \!fn void DoolittleLUPDecompose(Real **a, int n, int *pivot)

// \brief perform LU decomposition with partial (row) pivoting using Doolittle's
// algorithm. Partial pivoting is required for stability.
//
// Let D be a diagonal matrix, L be a unit lower triangular matrix (main diagonal is all
// 1's), and U be a unit upper triangular matrix
// Crout = (LD)U  ---> unit upper triangular U and L'=LD non-unit lower triangular
// Doolittle = L(DU) ---> unit lower triangular L and U'=DU non-unit upper triangular
//
// INPUT:
//     a: square nxn matrix A of real numbers. Must be a mutable pointer-to-pointer/rows.
//     n: number of rows and columns in "a"
//
//    Also expects "const Real lu_tol >=0" file-scope variable to be defined = criterion
//    for detecting degenerate input "a" (or nearly-degenerate).
//
// OUTPUT:
//     a: modified in-place to contain both lower- and upper-triangular matrices L, U
//        as A <- L + U (the 1's on the diagonal of L are not stored) in the decomposition
//        PA=LU. See NR pg 50; even though they claim to use Crout, they are probably
//        use Doolittle. They assume unit diagonal in Lx=Pb forward substitution.
// pivot: nx1 int vector that is a sparse representation of the nxn permutation matrix P.
//        For each row/vector entry, the value = the column # of the nonzero pivot element
//
// RETURN:
//  failure=0: routine detected that "a" matrix was nearly-singular
//  success=1: LUP decomposition completed
//
//     Both "a", "pivot" can then be passed with RHS vector "b" to DoolittleLUPSolve in
//     order to solve Ax=b system of linear equations
//
// REFERENCES:
//   - References Numerical Recipes, 3rd ed. (NR) section 2.3 "LU Decomposition & its
//     Applications"


int DoolittleLUPDecompose(Real **a, int n, int *pivot) {
  constexpr int failure = 0, success = 1;
  // initialize unit permutation matrix P=I. In our sparse representation, pivot[n]=n
  for (int i=0; i<=n; i++)
    pivot[i] = i;

  // loop over rows of input matrix:
  for (int i=0; i<n; i++) {
    Real a_max = 0.0, a_abs = 0.0;
    int i_max = i;
    // search for largest pivot element, located at row i_max
    for (int k=i; k<n; k++) {
      a_abs = std::abs(a[k][i]);
      if (a_abs > a_max) { // found larger pivot element
        a_max = a_abs;
        i_max = k;
      }
    }

    // if the pivot element is near zero, the matrix is likely singular
    if (a_max < lu_tol) {  // 0.0) { // see NR comment in ludcmp.h
      // do not divide by 0
      std::cout << std::scientific
                << std::setprecision(std::numeric_limits<Real>::max_digits10 -1)
                << "DoolittleLUPDecompose detects singular matrix with\n"
                << "pivot element=" << a_max << " < tolerance=" << lu_tol << std::endl;
      return failure;
    }

    if (i != i_max) {  // need to pivot rows:
      // pivoting "pivot" vector
      int row_idx = pivot[i];
      pivot[i] = pivot[i_max];
      pivot[i_max] = row_idx;

      // pivoting rows of A
      Real *pivot_ptr = a[i];
      a[i] = a[i_max];
      a[i_max] = pivot_ptr;
    }

    // these lines are the only difference from Crout's in-place approach w/ pivoting
    for (int j=i+1; j<n; j++) { // loop over rows; NR has the same approach as here
      // fill lower triangular matrix L elements at column "i":
      a[j][i] /= a[i][i];
      // (Crout finds upper triangular matrix U elemens at row "i" in this step)
      for (int k=i+1; k<n; k++) // update remaining submatrix
        a[j][k] -= a[j][i]*a[i][k];
    }
  }
  // in-place LU factorization with partial pivoting is complete
  return success;
}


//----------------------------------------------------------------------------------------
// \!fn void DoolittleLUPSolve(Real **lu, int *pivot, Real *b, int n, Real *x)

// \brief after DoolittleLUPDecompose() function has transformed input the LHS of Ax=b
// system to partially-row pivoted, LUP decomposed equivalent PAx=LUx=Pb, solve for x
//
// INPUT:
//     lu: square nxn matrix of real numbers containing output "a" of successful
//          DoolittleLUPDecompose() function call. See notes in that function for details.
//  pivot: nx1 vector of integers produced by DoolittleLUPDecompose()
//      b: RHS column vector of real numbers in original Ax=b system of linear equations
//
// OUTPUT:
//     x: nx1 column vector of real numbers containing solution in original Ax=b system

void DoolittleLUPSolve(Real **lu, int *pivot, Real *b, int n, Real *x) {
  // forward substitution, Ly=Pb (L must be a UNIT lower-triangular matrix)
  for (int i=0; i<n; i++) {
    // initialize the solution to the RHS values, repeating permutation from LU decomp.
    x[i] = b[pivot[i]];
    for (int j=0; j<i; j++)
      x[i] -= lu[i][j]*x[j];
  }

  // back substitution, Ux=y (U is a NOT a unit upper-triangular matrix)
  for (int i=(n-1); i>=0; i--) {
    for (int j=(i+1); j<n; j++) {
      x[i] -= x[j]*lu[i][j];
    }
    x[i] /= lu[i][i];
  }
  return;
}
} // namespace
}
