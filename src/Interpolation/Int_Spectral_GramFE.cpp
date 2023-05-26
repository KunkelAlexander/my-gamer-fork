#include "GAMER.h"


#if ( WAVE_SCHEME == WAVE_GRAMFE )
#include "GramExtensionTables.h"
#include "FFTW.h"

#include <complex.h>

using complex_type = std::complex<gramfe_float>;


void Int_ComputePeriodicExtension   ( gramfe_float* Input, int InputSize, gramfe_float* Ae, gramfe_float* Ao );
void Int_ComputeGramFEInterpolation ( gramfe_float* Input, int InputSize, gramfe_float* Ae, gramfe_float* Ao, complex_type* LCoeff, complex_type* RCoeff, gramfe_float* Output_LeftTranslated, gramfe_float* Output_RightTranslated, gramfe_real_fftw_plan f, gramfe_real_fftw_plan b1, gramfe_real_fftw_plan b2 );


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
void Int_Spectral( real CData[], const int CSize[3], const int CStart[3], const int CRange[3],
                    real FData[], const int FSize[3], const int FStart[3], const int NComp,
                    const bool UnwrapPhase, const bool Monotonic[], const real MonoCoeff, const bool OppSign0thOrder )
{

// interpolation-scheme-dependent parameters
// ===============================================================================
// number of coarse-grid ghost zone
   const int CGhost    = SPECTRAL_INT_GHOST_SIZE;
   const int InputSize = PS1 + 2 * CGhost;
// ===============================================================================

   //printf("Celebrating the spectral party!\n");

   const gramfe_float Norm = 1.0 / ( (gramfe_float) GRAMFE_CSIZE );
   gramfe_float K;
   complex_type LCoeff[GRAMFE_CSIZE/2 + 1], RCoeff[GRAMFE_CSIZE/2 + 1];     // exp(1j * dT * k^2)

// set up translation arrays
   for (int i=0; i<GRAMFE_CSIZE/2 + 1; i++)
   {
      //K         = ( i <= GRAMFE_CSIZE/2 ) ? 2.0*M_PI/(GRAMFE_CSIZE)*i : 2.0*M_PI/(GRAMFE_CSIZE)*(i-GRAMFE_CSIZE);
      K         = 2.0*M_PI/GRAMFE_CSIZE*i;
      LCoeff[i] = complex_type(COS(-K/4), SIN(-K/4)) * Norm;
      RCoeff[i] = complex_type(COS( K/4), SIN( K/4)) * Norm;
   }

   gramfe_float* Input, *Output_LeftTranslated, *Output_RightTranslated;
   Input                  = (gramfe_float*) gramfe_fftw_malloc( 2 * (GRAMFE_CSIZE / 2 + 1) * sizeof(gramfe_float) * 2);
   Output_LeftTranslated  = (gramfe_float*) gramfe_fftw_malloc( 2 * (GRAMFE_CSIZE / 2 + 1) * sizeof(gramfe_float) * 2);
   Output_RightTranslated = (gramfe_float*) gramfe_fftw_malloc( 2 * (GRAMFE_CSIZE / 2 + 1) * sizeof(gramfe_float) * 2);
   gramfe_float Ae[GRAMFE_NDELTA];
   gramfe_float Ao[GRAMFE_NDELTA];

   gramfe_real_fftw_plan forwardPlan  = gramfe_float_fftw3_plan_dft_r2c_1d( GRAMFE_CSIZE, (gramfe_float*)               Input,                  (gramfe_float_complex*)         Input,                  FFTW_ESTIMATE );
   gramfe_real_fftw_plan b1Plan       = gramfe_float_fftw3_plan_dft_c2r_1d( GRAMFE_CSIZE, (gramfe_float_complex*)       Output_LeftTranslated,  (gramfe_float*)                 Output_LeftTranslated,  FFTW_ESTIMATE );
   gramfe_real_fftw_plan b2Plan       = gramfe_float_fftw3_plan_dft_c2r_1d( GRAMFE_CSIZE, (gramfe_float_complex*)       Output_RightTranslated, (gramfe_float*)                 Output_RightTranslated, FFTW_ESTIMATE );

   for ( int i = 0; i < 3; ++i) {
      //printf("i = %d Csize[i] = %d Cstart[i] = %d Crange[i] = %d Fsize[i] = %d Fstart[i] = %d\n", i, CSize[i], CStart[i], CRange[i], FSize[i], FStart[i]);
   }

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

//       interpolate data using Gram Fourier extension
         Int_ComputeGramFEInterpolation(Input, InputSize, Ae, Ao, LCoeff, RCoeff, Output_LeftTranslated, Output_RightTranslated, forwardPlan, b1Plan, b2Plan);

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

//       interpolate data using Gram Fourier extension
         Int_ComputeGramFEInterpolation(Input, InputSize, Ae, Ao, LCoeff, RCoeff, Output_LeftTranslated, Output_RightTranslated, forwardPlan, b1Plan, b2Plan);

//       write result of Fourier interpolation to temporary array
         for (int In_y=CGhost, Out_y=0;  In_y<CGhost+CRange[1];   In_y++, Out_y+=2) {

            Idx_Out  = InOut_z*TdzY + Out_y*Tdy + InOut_x*Tdx;
            TDataY[ Idx_Out       ] = Output_LeftTranslated [In_y];
            TDataY[ Idx_Out + Tdy ] = Output_RightTranslated[In_y];
            //printf("TDataY Idx_Out %d Idx_Out + Tdy %d l %f r %f\n", Idx_Out      ,  Idx_Out + Tdy , TDataY[ Idx_Out       ], TDataY[ Idx_Out + Tdy ]);

         } // j
      } // k,i

      //printf("when would we like to dump the core? After y?\n");

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

//       interpolate data using Gram Fourier extension
         Int_ComputeGramFEInterpolation(Input, InputSize, Ae, Ao, LCoeff, RCoeff, Output_LeftTranslated, Output_RightTranslated, forwardPlan, b1Plan, b2Plan);

         for (int In_z=CGhost, Out_z=FStart[2];  In_z<CGhost+CRange[2];  In_z++, Out_z+=2)
         {
            Idx_Out = Out_z*Fdz  + Out_y*Fdy + Out_x*Fdx;

            FPtr[ Idx_Out       ] = Output_LeftTranslated  [In_z];
            FPtr[ Idx_Out + Fdz ] = Output_RightTranslated [In_z];
            //printf("FPtr Idx_Out %d Idx_Out + Fdz %d l %f r %f\n", Idx_Out, Idx_Out + Fdz, FPtr[ Idx_Out       ], FPtr[ Idx_Out + Fdz ]);

         }
      } // k,j,i
      //printf("when would we like to dump the core? After z?\n");


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
   gramfe_fftw_free( Input );
   gramfe_fftw_free( Output_LeftTranslated  );
   gramfe_fftw_free( Output_RightTranslated );

   gramfe_float_fftw3_destroy_plan(forwardPlan);
   gramfe_float_fftw3_destroy_plan(b1Plan);
   gramfe_float_fftw3_destroy_plan(b2Plan);

   //printf("when would we like to dump the core? After freeing data?\n");
} // FUNCTION : Int_Spectral

void Int_ComputePeriodicExtension (gramfe_float* Input, int InputSize, gramfe_float* Ae, gramfe_float* Ao) {
   gramfe_float Al, Ar;
// 2.1 compute Gram-Polynomial expansion coefficients via semi-discrete scalar products on boundary
   for (int si = 0; si < GRAMFE_ORDER; ++si )
   {
      Al = 0;
      Ar = 0;
      for (int t=0; t < GRAMFE_NDELTA; t++) {
         Al += Pl[si][t] * Input[t];                              // left boundary
         Ar += Pr[si][t] * Input[InputSize - GRAMFE_NDELTA + t]; // right boundary
      } // for t

      //printf("Al %f Ar %f\n", Al, Ar);

      Ae[si] = (gramfe_float) 0.5 * (Ar + Al);
      Ao[si] = (gramfe_float) 0.5 * (Ar - Al);
   } // for (int si = 0; si < GRAMFE_ORDER; ++si )

// 2.2 function values in extension domain given as linear combinations of extended Gram polynomials
   for (int si = InputSize; si < GRAMFE_CSIZE; ++si)
   {
      Input[si] = 0;

      for (int order = 0; order < GRAMFE_ORDER; order++) {
         Input[si] += Ae[order] *  Fe[order][si - InputSize];
         Input[si] += Ao[order] *  Fo[order][si - InputSize];
      } // for (int order=0; order < GRAMFE_ORDER; order++)
   } // for (int si = FLU_NXT; si < GRAMFE_CSIZE; ++si)

   for (int si = 0; si < GRAMFE_CSIZE; ++si) {
      //printf("si %d extended input %f\n", si, Input[si]);
   }
} // FUNCTION: Int_ComputePeriodicExtension

void Int_ComputeGramFEInterpolation ( gramfe_float* Input, int InputSize, gramfe_float* Ae, gramfe_float* Ao, complex_type* LCoeff, complex_type* RCoeff, gramfe_float* Output_LeftTranslated, gramfe_float* Output_RightTranslated, gramfe_real_fftw_plan f, gramfe_real_fftw_plan b1, gramfe_real_fftw_plan b2 ) {

/*
// interpolation coefficients
      const real R[ 3 ] = { -3.0/32.0, +30.0/32.0, +5.0/32.0 };
      const real L[ 3 ] = { +5.0/32.0, +30.0/32.0, -3.0/32.0 };
      //printf("We do the interpolation thing with an input size of %d!\n", InputSize);
      for (int i=1;  i < InputSize - 1; ++i)
      {
         Output_LeftTranslated [i] = L[0]*Input[i-1] + L[1]*Input[i] + L[2]*Input[i+1];
         Output_RightTranslated[i] = R[0]*Input[i-1] + R[1]*Input[i] + R[2]*Input[i+1];

         //printf("Interpolated values: l %f r %f \n", Output_LeftTranslated[i], Output_RightTranslated[i]);
      }
*/
//    compute Gram extension
      Int_ComputePeriodicExtension(Input, InputSize, Ae, Ao);

//    compute forward FFT
      gramfe_fftw_r2c( f, Input );

      complex_type* InputK   = (complex_type* ) Input;
      complex_type* OutputLK = (complex_type* ) Output_LeftTranslated;
      complex_type* OutputRK = (complex_type* ) Output_RightTranslated;

//    interpolate by shifting samples to the left by 0.25 dh
      for (int si = 0; si < GRAMFE_CSIZE/2 + 1; si++)
      {
         //printf("Input after FFT: %f + i %f\n", InputK[si].real(), InputK[si].imag());
         OutputLK[si] = InputK[si] * LCoeff[si];
         OutputRK[si] = InputK[si] * RCoeff[si];
      }
//    compute backward FFT
      gramfe_fftw_c2r( b1, OutputLK, Output_LeftTranslated );


//    compute backward FFT
      gramfe_fftw_c2r( b2, OutputRK, Output_RightTranslated );


      for (int si = 0; si < GRAMFE_CSIZE; si++)
      {
         //printf("Output %d in the center and the left and right: %f %f\n", si, Output_LeftTranslated[si], Output_RightTranslated[si]);
      }
} // FUNCTION: Int_ComputeGramFEInterpolation

#else
void Int_Spectral( real CData[], const int CSize[3], const int CStart[3], const int CRange[3],
                    real FData[], const int FSize[3], const int FStart[3], const int NComp,
                    const bool UnwrapPhase, const bool Monotonic[], const real MonoCoeff, const bool OppSign0thOrder )
{
}
#endif