#include "GAMER.h"

#if ( MODEL == ELBDM )

//-------------------------------------------------------------------------------------------------------
// Function    :  ELBDM_Flag_Interference
// Description :  Flag according to the interference criterion 
//
// Note        :  1. Flag the input cell if the interference criteria are met (Interference, Phase jumps and additional check whether dB wavelength is resolved)
//                2. Size of the input array "Cond_Array" should be PATCH_SIZE^3 
//
// Parameter   :  i,j,k       : Indices of the target cell in the arrays "Cond_Array"
//                Theshold    : Refinement Threshold for M_int
//
// Return      :  "true"  if the flag criterion is     fulfilled
//                "false" if the flag criterion is NOT fulfilled
//-------------------------------------------------------------------------------------------------------
bool ELBDM_Flag_Interference( const int i, const int j, const int k, const real Cond_Array[], const double Threshold)
{

// check
#  ifdef GAMER_DEBUG
   if (  i < 0  ||  i >= PS1  ||  j < 0  ||  j >= PS1  ||  k < 0  ||  k >= PS1  )
      Aux_Error( ERROR_INFO, "incorrect index (i,j,k) = (%d,%d,%d) !!\n", i, j, k );
#  endif

   const int    Idx       = k*PS1*PS1 + j*PS1 + i;
   const real   InterferenceCriterion   = Cond_Array[Idx];
   bool Flag = InterferenceCriterion > Threshold;

   return Flag;
} // FUNCTION : ELBDM_Flag_Interference




//-------------------------------------------------------------------------------------------------------
// Function    :  Prepare_for_Interference_Criterion
// Description :  Evaluate quantum pressure, phase jumps and whether dB wavelength will be resolved after refinement for the interference criterium
//
// Note        :  1. This function is called in "Flag_Real" before looping over all cells in the patch in order to
//                   achieve higher performance
//                2. Evaluate laplacian with second-order stencil and dB wavelength via first-order stencil
//                3. Do not take into account the physical size of each cell since criteria are dimensionless
//                4. The sizes of the arrays (Var1D, Temp1D, Cond1D) must be ( (PS1+2)^3, (PS1+2)^3, 3*(PS1)^3 )
//
// Parameter   :  Var       : Array storing the input density and phase/re & im for the interference criterion
//                Temp      : Array to store the intermediate variable sqrt(density)
//                Cond      : Array to store the output dimensionless quantum pressure, the curvature of the phase field as well as 
//                            the maximum phase difference between neighbouring points
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Prepare_for_Interference_Criterion(const real *Var1D, real *Temp1D, real *Cond1D, bool useWaveFlag)
{

   const int NCell  = PS1 + 2;   // size of the arrays Var, Temp
   const int NCond  = PS1;       // size of the array  Cond
   const float vortexThreshold = 0.5; 

   int ii, jj, kk, iim, jjm, kkm, iip, jjp, kkp;

// convert the 1D arrays
   real (*Var)  [NCell ][NCell ][NCell ] = ( real(*) [NCell ][NCell ][NCell ] )  Var1D;
   real (*Temp) [NCell ][NCell ][NCell ] = ( real(*) [NCell ][NCell ][NCell ] )  Temp1D;
   real (*Cond) [NCond ][NCond ][NCond ] = ( real(*) [NCond ][NCond ][NCond ] )  Cond1D;

   for (int k=0; k<NCell; k++)    {
   for (int j=0; j<NCell; j++)    {
   for (int i=0; i<NCell; i++)    {
      //Compute square root of density field
      Temp[0][k][j][i] = SQRT(Var[0][k][j][i]);

      //Check whether we are on fluid level and convert imaginary and real parts to phase accordingly
      if ( useWaveFlag ) {
         Temp[1][k][j][i] = SATAN2(Var[IMAG][k][j][i], Var[REAL][k][j][i]);
      } else {
         Temp[1][k][j][i] = Var[1][k][j][i];
      }
   }}} // k,j,i

   for (int k=0; k<NCond; k++)    {  kk = k + 1;   kkp = kk + 1;   kkm = kk - 1;
   for (int j=0; j<NCond; j++)    {  jj = j + 1;   jjp = jj + 1;   jjm = jj - 1;
   for (int i=0; i<NCond; i++)    {  ii = i + 1;   iip = ii + 1;   iim = ii - 1;

      //Compute the dimensionless quantum pressure (divided by number of dimensions for normalisation)
      Cond[0][k][j][i] =  FABS(  Temp[0][kk ][jj ][iip] + Temp[0][kk ][jj ][iim] \
                               + Temp[0][kk ][jjp][ii ] + Temp[0][kk ][jjm][ii ] \
                               + Temp[0][kkp][jj ][ii ] + Temp[0][kkm][jj ][ii ] \
                               -  (real) 6.0 * Temp[0][kk ][jj ][ii])\
                               / ((real) 3.0 * Temp[0][kk ][jj ][ii]);   
      if (useWaveFlag)   {
         Cond[1][k][j][i] = 0;

//       Resolve de Broglie wavelength when vortices are absent
//       Use quantum pressure combined with vortex threshold to determine whether vortex is present
         if ( Cond[0][k][j][i] < vortexThreshold ) {
            Cond[2][k][j][i] = MAX(MAX(MAX(MAX(MAX(
               FABS(Temp[1][kk ][jj ][iip] - ELBDM_UnwrapPhase(Temp[1][kk ][jj ][iip], Temp[1][kk ][jj ][ii ])),
               FABS(Temp[1][kk ][jj ][ii ] - ELBDM_UnwrapPhase(Temp[1][kk ][jj ][ii ], Temp[1][kk ][jj ][iim]))),
               FABS(Temp[1][kk ][jjp][ii ] - ELBDM_UnwrapPhase(Temp[1][kk ][jjp][ii ], Temp[1][kk ][jj ][ii ]))),
               FABS(Temp[1][kk ][jj ][ii ] - ELBDM_UnwrapPhase(Temp[1][kk ][jj ][ii ], Temp[1][kk ][jjm][ii ]))),
               FABS(Temp[1][kkp][jj ][ii ] - ELBDM_UnwrapPhase(Temp[1][kkp][jj ][ii ], Temp[1][kk ][jj ][ii ]))),
               FABS(Temp[1][kk ][jj ][ii ] - ELBDM_UnwrapPhase(Temp[1][kk ][jj ][ii ], Temp[1][kkm][jj ][ii ])));
         } else {
            Cond[2][k][j][i] = 0;
         }
      } else {

#        ifndef SMOOTH_PHASE
//       Check second derivative of phase field (divided by number of dimensions for normalisation) to detect phase jumps
         Cond[1][k][j][i] =  FABS( Temp[1][kk ][jj ][iip] + Temp[1][kk ][jj ][iim] \
                                 + Temp[1][kk ][jjp][ii ] + Temp[1][kk ][jjm][ii ] \
                                 + Temp[1][kkp][jj ][ii ] + Temp[1][kkm][jj ][ii ] \
                                 -  (real) 6.0 * Temp[1][kk ][jj ][ii])\
                                 / ((real) 3.0);
#        else    
         Cond[1][k][j][i] = 0;
#        endif                               

#        ifdef GAMER_DEBUG                       
//       Check maximum phase difference in all 6 directions
         Cond[2][k][j][i] = MAX(MAX(MAX(MAX(MAX(
            FABS(Temp[1][kk ][jj ][iip] - Temp[1][kk ][jj ][ii ]),
            FABS(Temp[1][kk ][jj ][ii ] - Temp[1][kk ][jj ][iim])),
            FABS(Temp[1][kk ][jjp][ii ] - Temp[1][kk ][jj ][ii ])),
            FABS(Temp[1][kk ][jj ][ii ] - Temp[1][kk ][jjm][ii ])),
            FABS(Temp[1][kkp][jj ][ii ] - Temp[1][kk ][jj ][ii ])),
            FABS(Temp[1][kk ][jj ][ii ] - Temp[1][kkm][jj ][ii ]));
#        else
         Cond[2][k][j][i] = 0;
#        endif 
      }
   }}} // k,j,i


} // FUNCTION : Prepare_for_Interference_Criterion



#endif // #if ( MODEL == ELBDM )
