#include "CUAPI.h"
#include "CUDA_ConstMemory.h"

#if ( defined GPU  &&  defined GRAVITY )


#ifdef GREP
extern void CUAPI_SetConstMemory_ExtPot_GREP();
#endif




//-------------------------------------------------------------------------------------------------------
// Function    :  CUAPI_SetConstMemory_ExtAccPot
// Description :  Set the constant memory variables on GPU used by the external acceleration and
//                potential routines
//
// Note        :  1. Adopt the suggested approach for CUDA version >= 5.0
//                2. Invoked by CUAPI_SetConstMemory() and EvolveLevel()
//                3. EXT_ACC_NAUX_MAX and EXT_POT_NAUX_MAX are defined in Macro.h
//
// Parameter   :  None
//
// Return      :  c_ExtAcc_AuxArray[], c_ExtPot_AuxArray[]
//---------------------------------------------------------------------------------------------------
void CUAPI_SetConstMemory_ExtAccPot()
{

   if ( OPT__EXT_ACC )
      CUDA_CHECK_ERROR(  cudaMemcpyToSymbol( c_ExtAcc_AuxArray, ExtAcc_AuxArray, EXT_ACC_NAUX_MAX*sizeof(double) )  );

   if ( OPT__EXT_POT )
      CUDA_CHECK_ERROR(  cudaMemcpyToSymbol( c_ExtPot_AuxArray, ExtPot_AuxArray, EXT_POT_NAUX_MAX*sizeof(double) )  );

#  ifdef GREP
   CUAPI_SetConstMemory_ExtPot_GREP();
#  endif

} // FUNCTION : CUAPI_SetConstMemory_ExtAccPot



#endif // #if ( defined GPU  &&  defined GRAVITY )
