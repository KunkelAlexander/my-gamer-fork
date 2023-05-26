#include "GAMER.h"


#if ( defined(SUPPORT_FFTW) && defined(SUPPORT_GSL) )


//gsl library
#  include <gsl/gsl_sf_gegenbauer.h>
#  include <gsl/gsl_complex_math.h>
#  include <gsl/gsl_permutation.h>
#  include <gsl/gsl_permute_vector.h>
#  include <gsl/gsl_permute_matrix.h>
#  include <gsl/gsl_linalg.h>


#include "FFTW.h"

#include <unordered_map>


class IPRContext {
public:
    IPRContext(size_t N, double lambda) : size(N), lambda(lambda) {

//      complex N^2 array
        changeOfBasisMatrix      = new double[2 * N * N];

//      real array with N polynomials of degree 0 - (N-1) evaluated at 2*N points
        interpolationPolynomials = new double[2 * (2 * N) * N];

//      evaluation matrix
        evaluationMatrix         = new double[2 * (2 * N) * N];

//      allocate memory for permutation
        p = gsl_permutation_alloc(N);

//      plan FFT of function to be interpolated
        fftw_complex* planner_array = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
        fftwPlan                    = create_gamer_fftw_1d_forward_c2c_plan(N, planner_array);
        fftw_free(planner_array);

//      compute W
        computeChangeOfBasisMatrix(changeOfBasisMatrix, N, lambda);

//      compute LU decomposition of W
        int signum;
        computeLUDecomposition(changeOfBasisMatrix, N, p1, &signum);

//      compute interpolation polynomials
        computeInterpolationPolynomials(lambda, N, interpolationPolynomials);

//
        computeEvaluationMatrix(evaluationMatrix, changeOfBasisMatrix, interpolationPolynomials, N, p2, &signum);


    }

    ~IPRContext() {
        delete [] changeOfBasisMatrix;
        delete [] interpolationPolynomials;
        delete [] evaluationMatrix;
        gsl_permutation_free(p);
        fftw_destroy_plan(fftwPlan);
    }

    double polynomial(int n, double lambda, double x) const
    {
        return gsl_sf_gegenpoly_n(n, lambda, x);
    }

    void computeChangeOfBasisMatrix(double* W, int N, double lambda) const
    {
        double*       input  = (double* )       fftw_malloc( N * 1 * sizeof( double )       );
        fftw_complex* output = (fftw_complex* ) W;

        for (int n = 0; n < N; ++n)
        {
//          evaluate polynomial in interval [-1, 1]
            for (int j = 0; j < N; ++j)
            {
                input[j] = polynomial( n, lambda, -1 + j / ((double)N / 2.0) );
            }

//          create FFTW plan for double-to-complex FFT
            fftw_plan plan = fftw_plan_dft_r2c_1d(N, input, &output[n * N], FFTW_ESTIMATE);

//          compute forward FFT
            fftw_execute(plan);

//          destroy the FFTW plan
            fftw_destroy_plan(plan);

//          real-to-complex FFT maps from n to n/2 +1 because of symmetry of real FFT
//          fill up remaining n - n/2 - 1 values with complex conjugate to obtain square matrix
            for (int j = (N / 2 + 1); j < N; ++j)
            {
                output[ n * N + j ][0] =   output[ n * N + N - j ][0];
                output[ n * N + j ][1] = - output[ n * N + N - j ][1];
            }
        }

//      transpose output array
        gsl_matrix_complex_view output_view = gsl_matrix_complex_view_array((double* ) output, N, N);
        gsl_matrix_complex_transpose(&output_view.matrix);

        fftw_free(input);
    }

    void computeLUDecomposition(double* input, size_t N, gsl_permutation* p, int* signum) {
        gsl_matrix_complex_view input_view = gsl_matrix_complex_view_array(input, N, N);

        gsl_linalg_complex_LU_decomp(&input_view.matrix, p, signum);
    }

    void computeEvaluationMatrix(double* transformationMatrix, const double* changeOfBasisMatrix, const double* interpolationPolynomials, size_t N, gsl_permutation* p, int* signum) {

        double* in         = new real[2 * N * N];
        double* out        = new real[2 * N * N];
        memcpy(in, changeOfBasisMatrix, 2* N * N * sizeof(double));

        gsl_matrix_complex_view input_view                    = gsl_matrix_complex_view_array(in,  N, N);
        gsl_matrix_complex_view output_view                   = gsl_matrix_complex_view_array(out, N, N);
        gsl_matrix_complex_const_view poly_view               = gsl_matrix_complex_const_view_array(interpolationPolynomials, 2 * N, N);
        gsl_matrix_complex_view       trans_view              = gsl_matrix_complex_view_array      (transformationMatrix,     2 * N, N);



//      set lower triangular part of matrix to zero
        for (size_t m = 1; m < N; ++m)
        {
            for (size_t n = 0; n < m; ++n)
            {
                gsl_matrix_complex_set(&input_view.matrix, m, n, {0, 0});
            }
        }

        gsl_linalg_complex_LU_decomp(&input_view.matrix, p, signum);


//      invert upper triangular part
        gsl_linalg_complex_LU_invert(&input_view.matrix, p, &output_view.matrix);

//      apply permutation
        //gsl_permute_matrix_complex(p, &output_view.matrix);


//      multiply inverted U by polynomials matrix to obtain transformation matrix
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, {1.0, 0.0}, &poly_view.matrix, &output_view.matrix, {0.0, 0.0}, &trans_view.matrix);

        delete[] in;
        delete[] out;
    }

    void computeInterpolationPolynomials(double lambda, size_t N, real *poly) const
    {
//      iterate over cells in interval [-1, 1] and evaluate polynomial
        for (size_t cell = 0; cell < 2 * N; ++cell)
        {
//      iterate over polynomials
            for (size_t polyOrder = 0; polyOrder < N; ++polyOrder)
            {
                poly[((cell * N) + polyOrder) * 2     ] = (real) polynomial(polyOrder, lambda, -1 + 1.0 / (2 * N) + (cell + 1 + 2 * (ghostBoundary - 1)) * 1.0 / N);
                poly[((cell * N) + polyOrder) * 2 + 1 ] = 0;
            }
        }
    }

    const double* getInterpolationPolynomials() const {
        return interpolationPolynomials;
    }

    const double* getChangeOfBasisMatrix() const {
        return changeOfBasisMatrix;
    }

    const double* getEvaluationMatrix() const {
        return evaluationMatrix;
    }

    fftw_plan getFFTWPlan() const {
        return fftwPlan;
    }

    const gsl_permutation* getFirstPermutation() const {
        return p;
    }

private:
    size_t                  size;
    double*                 changeOfBasisMatrix;
    double*                 interpolationPolynomials;
    double*                 evaluationMatrix;
    fftw_plan                fftwPlan;
    double                  lambda;
    gsl_permutation*        p;
};

class IPR {
public:
    enum InterpolationMode { InterpolateReal, InterpolateImag };

    void interpolateComplex(fftw_complex *input, fftw_complex *output, size_t N, size_t ghostBoundary) {

        if (contexts.find(N) == contexts.end()) {
            contexts.emplace(N, N);
        }

        IPRContext& c = contexts.at(N);

//      compute forward FFT
        gamer_fftw_c2c   (c.getFFTWPlan(), (fftw_complex *) input);

        size_t truncationN = N;

        gaussWithTruncation(c.getChangeOfBasisMatrix(), (double*) input, N, truncationN, IPR_TruncationThreshold, c.getFirstPermutation(),  c.getSecondPermutation());

        interpolateFunction(input, output, N, truncationN, ghostBoundary, c.getEvaluationMatrix() );
    }

    void interpolateReal(double* re_input, double *re_output, size_t N, size_t ghostBoundary) {

//      compute forward FFT
        fftw_complex* input  = (fftw_complex*) gamer_fftw_malloc(sizeof(fftw_complex) * N);
        fftw_complex* output = (fftw_complex*) gamer_fftw_malloc(sizeof(fftw_complex) * N * 2);

        for (size_t i = 0; i < N; ++i) {
            c_re(input[i]) = re_input[i];
            c_im(input[i]) = 0.0;
        }

        interpolateComplex(input, output, N, ghostBoundary);

        for (size_t i = 0; i < 2 * N - 2 * ghostBoundary; ++i) {
            re_output[i] = c_re(output[i]);
        }

        fftw_free(input);
        fftw_free(output);
    }

    void gaussWithTruncation(const double *LU, double *x, size_t N, size_t& truncationN, double truncationThreshold, const gsl_permutation* p) const
    {

        /*
        Solve Ax = B using Gaussian elimination and LU decomposition with truncation after the forward substitution for stability of IPR
        */

        // create matrix views for input vector and matrix
        gsl_matrix_complex_const_view A_view = gsl_matrix_complex_const_view_array(LU, N, N);
        const gsl_matrix_complex          *a = &A_view.matrix;

        gsl_vector_complex_view B_view = gsl_vector_complex_view_array(x, N);
        gsl_vector_complex          *b = &B_view.vector;


        // apply permutation p to b
        gsl_permute_vector_complex(p1, b);

        // forward substitution to solve for Ly = B
        gsl_blas_ztrsv(CblasLower, CblasNoTrans, CblasUnit, a, b);


        // truncation for IPR
        // necessary for convergence at large
        // ( see Short Note: On the numerical convergence with the inverse polynomial reconstruction method for the resolution of the Gibbs phenomenon, Jung and Shizgal 2007)
        for (size_t m = 0; m < N; ++m)
        {
            if (gsl_complex_abs(gsl_vector_complex_get(b, m)) < truncationThreshold)
            {
                //gsl_vector_complex_set(b, m, {0, 0});
                truncationN = m;
                break;
            }
        }

        // apply second permutation
        //gsl_permute_vector_complex(p2, b);

        //gsl_blas_ztrsv(CblasUpper, CblasNoTrans, CblasNonUnit, a, b);


    }

    void interpolateFunction(const fftw_complex* g, fftw_complex* output, size_t N, size_t truncationN, size_t ghostBoundary, const double* poly) const
    {

        gsl_matrix_complex_const_view A_view          = gsl_matrix_complex_const_view_array((double*) poly, 2 * N, N);
        gsl_matrix_complex_const_view truncatedA_view = gsl_matrix_complex_const_submatrix(&A_view.matrix, 0, 0, 2 * N, truncationN);
        const gsl_matrix_complex          *a = &truncatedA_view.matrix;

        gsl_vector_complex_view B_view = gsl_vector_complex_view_array((double*) g, truncationN);
        gsl_vector_complex          *b = &B_view.vector;

        gsl_vector_complex_view C_view = gsl_vector_complex_view_array((double*) output, 2 * N);
        gsl_vector_complex          *c = &C_view.vector;

        gsl_blas_zgemv(CblasNoTrans, {1.0, 0.0}, a, b, {0.0, 0.0}, c);

    } // FUNCTION : interpolateFunction


private:
    std::unordered_map<size_t, IPRContext> contexts;
};

IPR ipr;

//-------------------------------------------------------------------------------------------------------
// Function    :  Int_Spectral
// Description :  Perform spatial interpolation based on the Gram-Fourier extension method
//
// Note        :  1. The interpolation is spectrally accurate
//                2. The interpolation result is neither conservative nor monotonic
//                3. 3D interpolation is achieved by performing interpolation along x, y, and z directions
//                   in order
//
// Parameter   :  CData           : Input coarse-grid array
//                CSize           : Size of the CData array
//                CStart          : (x,y,z) starting indices to perform interpolation on the CData array
//                CRange          : Number of grids in each direction to perform interpolation
//                FData           : Output fine-grid array
//                FStart          : (x,y,z) starting indcies to store the interpolation results
//                NComp           : Number of components in the CData and FData array
//                UnwrapPhase     : Unwrap phase when OPT__INT_PHASE is on (for ELBDM only)
//                Monotonic       : Unused
//                MonoCoeff       : Unused
//                OppSign0thOrder : Unused
//-------------------------------------------------------------------------------------------------------
void Int_Spectral(  real CData[], const int CSize[3], const int CStart[3], const int CRange[3],
                    real FData[], const int FSize[3], const int FStart[3], const int NComp,
                    const bool UnwrapPhase, const bool Monotonic[], const real MonoCoeff, const bool OppSign0thOrder )
{

// interpolation-scheme-dependent parameters
// ===============================================================================
// number of coarse-grid ghost zone
   const int CGhost    = SPECTRAL_INT_GHOST_SIZE;
// ===============================================================================

   const int maxSize   = MAX(MAX(CSize[0], CSize[1]), CSize[2]) +  2 * CGhost;

   double* Input, *Output;
   Input  = (double*) fftw_malloc( 1 * maxSize * sizeof(double) * 2);
   Output = (double*) fftw_malloc( 2 * maxSize * sizeof(double) * 2);


// index stride of the coarse-grid input array
   const int Cdx    = 1;
   const int Cdy    = Cdx*CSize[0];
   const int Cdz    = Cdy*CSize[1];

// index stride of the temporary arrays storing the data after x and y interpolations
   const int Tdx    = 1;
   const int Tdy    = Tdx* CRange[0]*2;
   const int TdzX   = Tdy*(CRange[1]+2*CGhost);    // array after x interpolation
   const int TdzY   = Tdy* CRange[1]*2;            // array after y interpolation

// index stride of the fine-grid output array
   const int Fdx    = 1;
   const int Fdy    = Fdx*FSize[0];
   const int Fdz    = Fdy*FSize[1];

// index stride of different components
   const int CDisp  = CSize[0]*CSize[1]*CSize[2];
   const int FDisp  = FSize[0]*FSize[1]*FSize[2];


   real *CPtr   = CData;
   real *FPtr   = FData;
   real *TDataX = new real [ (CRange[2]+2*CGhost)*TdzX ];   // temporary array after x interpolation
   real *TDataY = new real [ (CRange[2]+2*CGhost)*TdzY ];   // temporary array after y interpolation



   int Idx_InL, Idx_InC, Idx_InR, Idx_Out;

#  if ( MODEL == ELBDM && ELBDM_SCHEME == ELBDM_HYBRID && defined(SMOOTH_PHASE) )

// index stride of the coarse-grid input array without ghost boundary (without ghost = woG)
   const int CwoGdx = 1;
   const int CwoGdy = CwoGdx*CRange[0];
   const int CwoGdz = CwoGdy*CRange[1];

   real *TData_GlobalPhase  = NULL;

   if ( UnwrapPhase == 2)
   {
      TData_GlobalPhase = new real [ CRange[0] * CRange[1] * CRange[2] ];


      for (int k=CStart[2];  k<CStart[2]+CRange[2];  k++)
      for (int j=CStart[1];  j<CStart[1]+CRange[1];  j++)
      for (int i=CStart[0];  i<CStart[0]+CRange[0];  i++)
      {
         Idx_InC      = k*Cdz + j*Cdy + i*Cdx;
         Idx_Out      = (k - CStart[2]) * CwoGdz + (j - CStart[1]) * CwoGdy + (i - CStart[0]) *CwoGdx;
         TData_GlobalPhase[Idx_Out] = CPtr[Idx_InC];
      }

   }
#  endif // #  if ( MODEL == ELBDM && ELBDM_SCHEME == ELBDM_HYBRID && defined(SMOOTH_PHASE) )

   for (int v=0; v<NComp; v++)
   {
      //printf("when would we like to dump the core? At the start?\n");
//    unwrap phase along x direction
#     if ( MODEL == ELBDM )
      if ( UnwrapPhase )
      {
         for (int k=CStart[2]-CGhost;    k<CStart[2]+CRange[2]+CGhost;  k++)
         for (int j=CStart[1]-CGhost;    j<CStart[1]+CRange[1]+CGhost;  j++)
         for (int i=CStart[0]-CGhost+1;  i<CStart[0]+CRange[0]+CGhost;  i++)
         {
            Idx_InC       = k*Cdz + j*Cdy + i*Cdx;
            Idx_InL       = Idx_InC - Cdx;
            //          only unwrap if we detect discontinuity
#           if ( MODEL == ELBDM && ELBDM_SCHEME == ELBDM_HYBRID && defined(SMOOTH_PHASE) )
            if ( Int_HasDiscontinuity(CPtr, Idx_InC, Cdx, i == CStart[0]+CRange[0]+CGhost - 1) )
#           endif // #  if ( MODEL == ELBDM && ELBDM_SCHEME == ELBDM_HYBRID && defined(SMOOTH_PHASE) )
            CPtr[Idx_InC] = ELBDM_UnwrapPhase( CPtr[Idx_InL], CPtr[Idx_InC] );
         }
      }
#     endif


//    interpolation along x direction
      for (int In_z=CStart[2]-CGhost, Out_z=0;  In_z<CStart[2]+CRange[2]+CGhost;  In_z++, Out_z++)
      for (int In_y=CStart[1]-CGhost, Out_y=0;  In_y<CStart[1]+CRange[1]+CGhost;  In_y++, Out_y++)
      {

//       fill FFT array with one column of input data (including ghost zones) in x-direction
         for (int In_x=CStart[0]-CGhost, Out_x=0;  In_x<CStart[0]+CRange[0]+CGhost;  In_x++, Out_x++)
         {
            Idx_InC      = In_z*Cdz  +  In_y*Cdy +  In_x*Cdx;
            Input[Out_x] = CPtr[Idx_InC];
         } // i

//       interpolate data using IPR
         ipr.interpolateReal(Input, Output, CRange[0]+ 2 * CGhost, CGhost);

//       write result of Fourier interpolation (excluding ghost zones) to temporary array
         for (int In_x=CGhost, Out_x=0; In_x<CRange[0]+CGhost; In_x++, Out_x+=2)
         {
            Idx_Out = Out_z*TdzX + Out_y*Tdy + Out_x*Tdx;

            TDataX[ Idx_Out       ] = Output_LeftTranslated [In_x];
            TDataX[ Idx_Out + Tdx ] = Output_RightTranslated[In_x];
            //printf("TDataX Idx_Out %d Idx_Out + Tdx %d l %f r %f\n", Idx_Out      ,  Idx_Out + Tdx , TDataX[ Idx_Out       ], TDataX[ Idx_Out + Tdx ]);


         } // i
      } // k,j

      //printf("when would we like to dump the core? After x?\n");

//    unwrap phase along y direction
#     if ( MODEL == ELBDM )
      if ( UnwrapPhase )
      {
         for (int k=0;  k<CRange[2]+2*CGhost;  k++)
         for (int j=1;  j<CRange[1]+2*CGhost;  j++)
         for (int i=0;  i<2*CRange[0];         i++)
         {
            Idx_InC         = k*TdzX + j*Tdy + i*Tdx;
            Idx_InL         = Idx_InC - Tdy;
//          only unwrap if we detect discontinuity
#           if ( MODEL == ELBDM && ELBDM_SCHEME == ELBDM_HYBRID && defined(SMOOTH_PHASE) )
            if ( Int_HasDiscontinuity(CPtr, Idx_InC, Tdy, j == CRange[1] + 2*CGhost - 1) )
#           endif // #  if ( MODEL == ELBDM && ELBDM_SCHEME == ELBDM_HYBRID && defined(SMOOTH_PHASE) )
            TDataX[Idx_InC] = ELBDM_UnwrapPhase( TDataX[Idx_InL], TDataX[Idx_InC] );
         }
      }
#     endif


//    interpolation along y direction
      for (int InOut_z=0;             InOut_z<CRange[2]+2*CGhost;  InOut_z++)
      for (int InOut_x=0;             InOut_x<2*CRange[0];         InOut_x++)
      {

//       fill FFT array with one column of input data (including ghost boundary) in y-direction
         for (int In_y=0, Out_y=0;    In_y < CRange[1]+2*CGhost;   In_y++, Out_y++) {
            Idx_InC      = InOut_z*TdzX +  In_y*Tdy + InOut_x*Tdx;
            Input[Out_y] = TDataX[Idx_InC];
         } // j

//       interpolate data using IPR
         ipr.interpolateReal(Input, Output, CRange[1]+ 2 * CGhost, CGhost);


//       write result of Fourier interpolation to temporary array
         for (int In_y=CGhost, Out_y=0;  In_y<CGhost+CRange[1];   In_y++, Out_y+=2) {

            Idx_Out  = InOut_z*TdzY + Out_y*Tdy + InOut_x*Tdx;
            TDataY[ Idx_Out       ] = Output_LeftTranslated [In_y];
            TDataY[ Idx_Out + Tdy ] = Output_RightTranslated[In_y];
            //printf("TDataY Idx_Out %d Idx_Out + Tdy %d l %f r %f\n", Idx_Out      ,  Idx_Out + Tdy , TDataY[ Idx_Out       ], TDataY[ Idx_Out + Tdy ]);

         } // j
      } // k,i


//    unwrap phase along z direction
#     if ( MODEL == ELBDM )
      if ( UnwrapPhase )
      {
         for (int k=1;  k<CRange[2]+2*CGhost;  k++)
         for (int j=0;  j<2*CRange[1];         j++)
         for (int i=0;  i<2*CRange[0];         i++)
         {
            Idx_InC         = k*TdzY + j*Tdy + i*Tdx;
            Idx_InL         = Idx_InC - TdzY;
//          only unwrap if we detect discontinuity
#           if ( MODEL == ELBDM && ELBDM_SCHEME == ELBDM_HYBRID && defined(SMOOTH_PHASE) )
            if ( Int_HasDiscontinuity( CPtr, Idx_InC, TdzY, k == CRange[2] + 2*CGhost - 1) )
#           endif // #  if ( MODEL == ELBDM && ELBDM_SCHEME == ELBDM_HYBRID && defined(SMOOTH_PHASE) )
            TDataY[Idx_InC] = ELBDM_UnwrapPhase( TDataY[Idx_InL], TDataY[Idx_InC] );
         }
      }
#     endif


//    interpolation along z direction
      for (int In_y=0,      Out_y=FStart[1];  In_y<2*CRange[1];       In_y++, Out_y++)
      for (int In_x=0,      Out_x=FStart[0];  In_x<2*CRange[0];       In_x++, Out_x++)
      {

//       fill FFT array with one column of input data in z-direction
         for (int In_z=0, Out_z=0;  In_z<CRange[2]+2*CGhost;  In_z++, Out_z++)
         {
            Idx_InC      = In_z*TdzY +  In_y*Tdy +  In_x*Tdx;
            Input[Out_z] = TDataY[Idx_InC];
         }

//       interpolate data using IPR
         ipr.interpolateReal(Input, Output, CRange[1]+ 2 * CGhost, CGhost);

         for (int In_z=CGhost, Out_z=FStart[2];  In_z<CGhost+CRange[2];  In_z++, Out_z+=2)
         {
            Idx_Out = Out_z*Fdz  + Out_y*Fdy + Out_x*Fdx;

            FPtr[ Idx_Out       ] = Output_LeftTranslated  [In_z];
            FPtr[ Idx_Out + Fdz ] = Output_RightTranslated [In_z];
            //printf("FPtr Idx_Out %d Idx_Out + Fdz %d l %f r %f\n", Idx_Out, Idx_Out + Fdz, FPtr[ Idx_Out       ], FPtr[ Idx_Out + Fdz ]);

         }
      } // k,j,i


#     if ( MODEL == ELBDM && ELBDM_SCHEME == ELBDM_HYBRID && defined(SMOOTH_PHASE) )

      if ( UnwrapPhase == 2)
      {
         real shift;

         for (int In_z=0, Out_z=0;  In_z<CRange[2];  In_z++, Out_z+=2)
         for (int In_y=0, Out_y=0;  In_y<CRange[1];  In_y++, Out_y+=2)
         for (int In_x=0, Out_x=0;  In_x<CRange[0];  In_x++, Out_x+=2)
         {
            Idx_InC       =  In_z * CwoGdz +  In_y * CwoGdy +  In_x * CwoGdx;
            Idx_Out       = Out_z *    Fdz + Out_y *    Fdy + Out_x *    Fdx;
            shift         = real(2 * M_PI) * ELBDM_UnwrapWindingNumber( TData_GlobalPhase[Idx_InC], FPtr[Idx_Out] );

            FPtr[Idx_Out             ]       += shift;
            FPtr[Idx_Out + Fdx       ]       += shift;
            FPtr[Idx_Out + Fdy       ]       += shift;
            FPtr[Idx_Out + Fdz       ]       += shift;
            FPtr[Idx_Out + Fdx + Fdy ]       += shift;
            FPtr[Idx_Out + Fdx + Fdz ]       += shift;
            FPtr[Idx_Out + Fdy + Fdz ]       += shift;
            FPtr[Idx_Out + Fdx + Fdy + Fdz]  += shift;
         }
      }
#     endif // #  if ( MODEL == ELBDM && ELBDM_SCHEME == ELBDM_HYBRID && defined(SMOOTH_PHASE) )

      CPtr += CDisp;
      FPtr += FDisp;

      //printf("when would we like to dump the core? After the first disp?\n");
   } // for (int v=0; v<NComp; v++)

#  if ( MODEL == ELBDM && ELBDM_SCHEME == ELBDM_HYBRID && defined(SMOOTH_PHASE) )
   if ( UnwrapPhase == 2) delete [] TData_GlobalPhase;
#  endif // ( MODEL == ELBDM && ELBDM_SCHEME == ELBDM_HYBRID && defined(SMOOTH_PHASE) )

   delete [] TDataX;
   delete [] TDataY;

   //printf("when would we like to dump the core? Before freeing data?\n");
   fftw_free( Input  );
   fftw_free( Output );

} // FUNCTION : Int_Spectral

#endif // #if ( defined(SUPPORT_FFTW) && defined(SUPPORT_GSL) )