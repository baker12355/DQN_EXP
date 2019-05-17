/* TA-LIB Copyright (c) 1999-2007, Mario Fortier
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in
 *   the documentation and/or other materials provided with the
 *   distribution.
 *
 * - Neither name of author nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* List of contributors:
 *
 *  Initial  Name/description
 *  -------------------------------------------------------------------
 *  MF       Mario Fortier
 *
 *
 * Change history:
 *
 *  MMDDYY BY   Description
 *  -------------------------------------------------------------------
 *  120802 MF   Template creation.
 *  023003 MF   Initial Coding of MAMA.
 *  052603 MF   Adapt code to compile with .NET Managed C++
 *
 */

/**** START GENCODE SECTION 1 - DO NOT DELETE THIS LINE ****/
/* All code within this section is automatically
 * generated by gen_code. Any modification will be lost
 * next time gen_code is run.
 */
/* Generated */ 
/* Generated */ #if defined( _MANAGED )
/* Generated */    #include "TA-Lib-Core.h"
/* Generated */    #define TA_INTERNAL_ERROR(Id) (RetCode::InternalError)
/* Generated */    namespace TicTacTec { namespace TA { namespace Library {
/* Generated */ #elif defined( _JAVA )
/* Generated */    #include "ta_defs.h"
/* Generated */    #include "ta_java_defs.h"
/* Generated */    #define TA_INTERNAL_ERROR(Id) (RetCode.InternalError)
/* Generated */ #else
/* Generated */    #include <string.h>
/* Generated */    #include <math.h>
/* Generated */    #include "ta_func.h"
/* Generated */ #endif
/* Generated */ 
/* Generated */ #ifndef TA_UTILITY_H
/* Generated */    #include "ta_utility.h"
/* Generated */ #endif
/* Generated */ 
/* Generated */ #ifndef TA_MEMORY_H
/* Generated */    #include "ta_memory.h"
/* Generated */ #endif
/* Generated */ 
/* Generated */ #define TA_PREFIX(x) TA_##x
/* Generated */ #define INPUT_TYPE   double
/* Generated */ 
/* Generated */ #if defined( _MANAGED )
/* Generated */ int Core::MamaLookback( double        optInFastLimit, /* From 0.01 to 0.99 */
/* Generated */                       double        optInSlowLimit )  /* From 0.01 to 0.99 */
/* Generated */ 
/* Generated */ #elif defined( _JAVA )
/* Generated */ public int mamaLookback( double        optInFastLimit, /* From 0.01 to 0.99 */
/* Generated */                        double        optInSlowLimit )  /* From 0.01 to 0.99 */
/* Generated */ 
/* Generated */ #else
/* Generated */ int TA_MAMA_Lookback( double        optInFastLimit, /* From 0.01 to 0.99 */
/* Generated */                     double        optInSlowLimit )  /* From 0.01 to 0.99 */
/* Generated */ 
/* Generated */ #endif
/**** END GENCODE SECTION 1 - DO NOT DELETE THIS LINE ****/
{
   /* insert local variable here */

/**** START GENCODE SECTION 2 - DO NOT DELETE THIS LINE ****/
/* Generated */ #ifndef TA_FUNC_NO_RANGE_CHECK
/* Generated */    if( optInFastLimit == TA_REAL_DEFAULT )
/* Generated */       optInFastLimit = 5.000000e-1;
/* Generated */    else if( (optInFastLimit < 1.000000e-2) ||/* Generated */  (optInFastLimit > 9.900000e-1) )
/* Generated */       return -1;
/* Generated */ 
/* Generated */    if( optInSlowLimit == TA_REAL_DEFAULT )
/* Generated */       optInSlowLimit = 5.000000e-2;
/* Generated */    else if( (optInSlowLimit < 1.000000e-2) ||/* Generated */  (optInSlowLimit > 9.900000e-1) )
/* Generated */       return -1;
/* Generated */ 
/* Generated */ #endif /* TA_FUNC_NO_RANGE_CHECK */
/**** END GENCODE SECTION 2 - DO NOT DELETE THIS LINE ****/

   /* insert lookback code here. */

   /* The two parameters are not a factor to determine
    * the lookback, but are still requested for 
    * consistency with all other Lookback functions.
    */
   UNUSED_VARIABLE(optInFastLimit);
   UNUSED_VARIABLE(optInSlowLimit);

   /* Lookback is a fix amount + the unstable period.
    *
    *
    * The fix lookback is 32 and is establish as follow:
    *    
    *         12 price bar to be compatible with the implementation
    *            of TradeStation found in John Ehlers book.
    *          6 price bars for the Detrender
    *          6 price bars for Q1
    *          3 price bars for jI
    *          3 price bars for jQ
    *          1 price bar for Re/Im
    *          1 price bar for the Delta Phase
    *        -------
    *         32 Total
    */

   return 32 + TA_GLOBALS_UNSTABLE_PERIOD(TA_FUNC_UNST_MAMA,Mama);
}

/**** START GENCODE SECTION 3 - DO NOT DELETE THIS LINE ****/
/*
 * TA_MAMA - MESA Adaptive Moving Average
 * 
 * Input  = double
 * Output = double, double
 * 
 * Optional Parameters
 * -------------------
 * optInFastLimit:(From 0.01 to 0.99)
 *    Upper limit use in the adaptive algorithm
 * 
 * optInSlowLimit:(From 0.01 to 0.99)
 *    Lower limit use in the adaptive algorithm
 * 
 * 
 */
/* Generated */ 
/* Generated */ #if defined( _MANAGED ) && defined( USE_SUBARRAY )
/* Generated */ enum class Core::RetCode Core::Mama( int    startIdx,
/* Generated */                                      int    endIdx,
/* Generated */                                      SubArray^    inReal,
/* Generated */                                      double        optInFastLimit, /* From 0.01 to 0.99 */
/* Generated */                                      double        optInSlowLimit, /* From 0.01 to 0.99 */
/* Generated */                                      [Out]int%    outBegIdx,
/* Generated */                                      [Out]int%    outNBElement,
/* Generated */                                      cli::array<double>^  outMAMA,
/* Generated */                                      cli::array<double>^  outFAMA )
/* Generated */ #elif defined( _MANAGED )
/* Generated */ enum class Core::RetCode Core::Mama( int    startIdx,
/* Generated */                                      int    endIdx,
/* Generated */                                      cli::array<double>^ inReal,
/* Generated */                                      double        optInFastLimit, /* From 0.01 to 0.99 */
/* Generated */                                      double        optInSlowLimit, /* From 0.01 to 0.99 */
/* Generated */                                      [Out]int%    outBegIdx,
/* Generated */                                      [Out]int%    outNBElement,
/* Generated */                                      cli::array<double>^  outMAMA,
/* Generated */                                      cli::array<double>^  outFAMA )
/* Generated */ #elif defined( _JAVA )
/* Generated */ public RetCode mama( int    startIdx,
/* Generated */                      int    endIdx,
/* Generated */                      double       inReal[],
/* Generated */                      double        optInFastLimit, /* From 0.01 to 0.99 */
/* Generated */                      double        optInSlowLimit, /* From 0.01 to 0.99 */
/* Generated */                      MInteger     outBegIdx,
/* Generated */                      MInteger     outNBElement,
/* Generated */                      double        outMAMA[],
/* Generated */                      double        outFAMA[] )
/* Generated */ #else
/* Generated */ TA_RetCode TA_MAMA( int    startIdx,
/* Generated */                     int    endIdx,
/* Generated */                     const double inReal[],
/* Generated */                     double        optInFastLimit, /* From 0.01 to 0.99 */
/* Generated */                     double        optInSlowLimit, /* From 0.01 to 0.99 */
/* Generated */                     int          *outBegIdx,
/* Generated */                     int          *outNBElement,
/* Generated */                     double        outMAMA[],
/* Generated */                     double        outFAMA[] )
/* Generated */ #endif
/**** END GENCODE SECTION 3 - DO NOT DELETE THIS LINE ****/
{
	/* insert local variable here */

   int outIdx, i;
   int lookbackTotal, today;
   double tempReal, tempReal2;

   double adjustedPrevPeriod, period;

   /* Variable used for the price smoother (a weighted moving average). */
   int trailingWMAIdx;
   double periodWMASum, periodWMASub, trailingWMAValue;
   double smoothedValue;

   /* Variables used for the Hilbert Transormation */
   CONSTANT_DOUBLE(a) = 0.0962;
   CONSTANT_DOUBLE(b) = 0.5769;
   double hilbertTempReal;
   int hilbertIdx;

   HILBERT_VARIABLES( detrender );
   HILBERT_VARIABLES( Q1 );
   HILBERT_VARIABLES( jI );
   HILBERT_VARIABLES( jQ );

   double Q2, I2, prevQ2, prevI2, Re, Im;

   double I1ForOddPrev2,  I1ForOddPrev3;
   double I1ForEvenPrev2, I1ForEvenPrev3;

   double rad2Deg;

   double mama,fama,todayValue,prevPhase;

/**** START GENCODE SECTION 4 - DO NOT DELETE THIS LINE ****/
/* Generated */ 
/* Generated */ #ifndef TA_FUNC_NO_RANGE_CHECK
/* Generated */ 
/* Generated */    /* Validate the requested output range. */
/* Generated */    if( startIdx < 0 )
/* Generated */       return ENUM_VALUE(RetCode,TA_OUT_OF_RANGE_START_INDEX,OutOfRangeStartIndex);
/* Generated */    if( (endIdx < 0) || (endIdx < startIdx))
/* Generated */       return ENUM_VALUE(RetCode,TA_OUT_OF_RANGE_END_INDEX,OutOfRangeEndIndex);
/* Generated */ 
/* Generated */    #if !defined(_JAVA)
/* Generated */    if( !inReal ) return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */    #endif /* !defined(_JAVA)*/
/* Generated */    if( optInFastLimit == TA_REAL_DEFAULT )
/* Generated */       optInFastLimit = 5.000000e-1;
/* Generated */    else if( (optInFastLimit < 1.000000e-2) ||/* Generated */  (optInFastLimit > 9.900000e-1) )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    if( optInSlowLimit == TA_REAL_DEFAULT )
/* Generated */       optInSlowLimit = 5.000000e-2;
/* Generated */    else if( (optInSlowLimit < 1.000000e-2) ||/* Generated */  (optInSlowLimit > 9.900000e-1) )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    #if !defined(_JAVA)
/* Generated */    if( !outMAMA )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    if( !outFAMA )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    #endif /* !defined(_JAVA) */
/* Generated */ #endif /* TA_FUNC_NO_RANGE_CHECK */
/* Generated */ 
/**** END GENCODE SECTION 4 - DO NOT DELETE THIS LINE ****/

   /* Insert TA function code here. */

   /* Constant */
   rad2Deg = 180.0 / (4.0 * std_atan(1));

   /* Identify the minimum number of price bar needed
    * to calculate at least one output.
    */
   lookbackTotal = 32 + TA_GLOBALS_UNSTABLE_PERIOD(TA_FUNC_UNST_MAMA,Mama);

   /* Move up the start index if there is not
    * enough initial data.
    */
   if( startIdx < lookbackTotal )
      startIdx = lookbackTotal;

   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
      VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
      return ENUM_VALUE(RetCode,TA_SUCCESS,Success);
   }

   VALUE_HANDLE_DEREF(outBegIdx) = startIdx;

   /* Initialize the price smoother, which is simply a weighted
    * moving average of the price.
    * To understand this algorithm, I strongly suggest to understand
    * first how TA_WMA is done.
    */
   trailingWMAIdx = startIdx - lookbackTotal;
   today = trailingWMAIdx;

   /* Initialization is same as WMA, except loop is unrolled
    * for speed optimization.
    */
   tempReal = inReal[today++];
   periodWMASub = tempReal;
   periodWMASum = tempReal;
   tempReal = inReal[today++];
   periodWMASub += tempReal;
   periodWMASum += tempReal*2.0;
   tempReal = inReal[today++];
   periodWMASub += tempReal;
   periodWMASum += tempReal*3.0;

   trailingWMAValue = 0.0;

   /* Subsequent WMA value are evaluated by using
    * the DO_PRICE_WMA macro.
    */
   #define DO_PRICE_WMA(varNewPrice,varToStoreSmoothedValue) { \
      periodWMASub     += varNewPrice; \
      periodWMASub     -= trailingWMAValue; \
      periodWMASum     += varNewPrice*4.0; \
      trailingWMAValue  = inReal[trailingWMAIdx++]; \
      varToStoreSmoothedValue = periodWMASum*0.1; \
      periodWMASum -= periodWMASub; \
   }

   i = 9;
   do
   {
      tempReal = inReal[today++];
      DO_PRICE_WMA(tempReal,smoothedValue);
   } while( --i != 0);

   /* Initialize the circular buffers used by the hilbert
    * transform logic. 
    * A buffer is used for odd day and another for even days.
    * This minimize the number of memory access and floating point
    * operations needed (note also that by using static circular buffer, 
    * no large dynamic memory allocation is needed for storing
    * intermediate calculation!).
    */
   hilbertIdx = 0;

   INIT_HILBERT_VARIABLES(detrender);
   INIT_HILBERT_VARIABLES(Q1);
   INIT_HILBERT_VARIABLES(jI);
   INIT_HILBERT_VARIABLES(jQ);

   period = 0.0;
   outIdx = 0;

   prevI2 = prevQ2 = 0.0;
   Re     = Im     = 0.0;
   mama   = fama   = 0.0;
   I1ForOddPrev3 = I1ForEvenPrev3 = 0.0;
   I1ForOddPrev2 = I1ForEvenPrev2 = 0.0;

   prevPhase  = 0.0;

   /* The code is speed optimized and is most likely very
    * hard to follow if you do not already know well the
    * original algorithm.
    * To understadn better, it is strongly suggested to look 
    * first at the Excel implementation in "test_MAMA.xls" included
    * in this package.
    */
   while( today <= endIdx )
   {
      adjustedPrevPeriod = (0.075*period)+0.54;

      todayValue = inReal[today];
      DO_PRICE_WMA(todayValue,smoothedValue);

      if( (today%2) == 0 )
      {
         /* Do the Hilbert Transforms for even price bar */
         DO_HILBERT_EVEN(detrender,smoothedValue);
         DO_HILBERT_EVEN(Q1,detrender);
         DO_HILBERT_EVEN(jI,I1ForEvenPrev3);
         DO_HILBERT_EVEN(jQ,Q1);
         if( ++hilbertIdx == 3 )
            hilbertIdx = 0;

         Q2 = (0.2*(Q1 + jI)) + (0.8*prevQ2);
         I2 = (0.2*(I1ForEvenPrev3 - jQ)) + (0.8*prevI2);

         /* The variable I1 is the detrender delayed for
          * 3 price bars. 
          *
          * Save the current detrender value for being
          * used by the "odd" logic later.
          */
         I1ForOddPrev3 = I1ForOddPrev2;
         I1ForOddPrev2 = detrender;

         /* Put Alpha in tempReal2 */
         if( I1ForEvenPrev3 != 0.0 )
            tempReal2 = (std_atan(Q1/I1ForEvenPrev3)*rad2Deg);
         else
            tempReal2 = 0.0;
      }
      else
      {
         /* Do the Hilbert Transforms for odd price bar */
         DO_HILBERT_ODD(detrender,smoothedValue);
         DO_HILBERT_ODD(Q1,detrender);
         DO_HILBERT_ODD(jI,I1ForOddPrev3);
         DO_HILBERT_ODD(jQ,Q1);

         Q2 = (0.2*(Q1 + jI)) + (0.8*prevQ2);
         I2 = (0.2*(I1ForOddPrev3 - jQ)) + (0.8*prevI2);

         /* The varaiable I1 is the detrender delayed for
          * 3 price bars. 
          *
          * Save the current detrender value for being
          * used by the "odd" logic later.
          */
         I1ForEvenPrev3 = I1ForEvenPrev2;
         I1ForEvenPrev2 = detrender;

         /* Put Alpha in tempReal2 */
         if( I1ForOddPrev3 != 0.0 )
            tempReal2 = (std_atan(Q1/I1ForOddPrev3)*rad2Deg);
         else
            tempReal2 = 0.0;
      }

      /* Put Delta Phase into tempReal */
      tempReal  = prevPhase - tempReal2;
      prevPhase = tempReal2;
      if( tempReal < 1.0 )
         tempReal = 1.0;

      /* Put Alpha into tempReal */
      if( tempReal > 1.0 )
      {
         tempReal = optInFastLimit/tempReal;
         if( tempReal < optInSlowLimit )
            tempReal = optInSlowLimit;
      }
      else
      {
         tempReal = optInFastLimit;
      }  

      /* Calculate MAMA, FAMA */
      mama = (tempReal*todayValue)+((1-tempReal)*mama);
      tempReal *= 0.5;
      fama = (tempReal*mama)+((1-tempReal)*fama);
      if( today >= startIdx )
      {
         outMAMA[outIdx] = mama;
         outFAMA[outIdx++] = fama;
      }

      /* Adjust the period for next price bar */
      Re = (0.2*((I2*prevI2)+(Q2*prevQ2)))+(0.8*Re);
      Im = (0.2*((I2*prevQ2)-(Q2*prevI2)))+(0.8*Im);
      prevQ2 = Q2;
      prevI2 = I2;
      tempReal = period;
      if( (Im != 0.0) && (Re != 0.0) )
         period = 360.0 / (std_atan(Im/Re)*rad2Deg);
      tempReal2 = 1.5*tempReal;
      if( period > tempReal2)
         period = tempReal2;
      tempReal2 = 0.67*tempReal;
      if( period < tempReal2 )
         period = tempReal2;
      if( period < 6 )
         period = 6;
      else if( period > 50 )
         period = 50;
      period = (0.2*period) + (0.8 * tempReal);

      /* Ooof... let's do the next price bar now! */
      today++;
   }

   /* Default return values */
   VALUE_HANDLE_DEREF(outNBElement) = outIdx;

   return ENUM_VALUE(RetCode,TA_SUCCESS,Success);
}

/**** START GENCODE SECTION 5 - DO NOT DELETE THIS LINE ****/
/* Generated */ 
/* Generated */ #define  USE_SINGLE_PRECISION_INPUT
/* Generated */ #if !defined( _MANAGED ) && !defined( _JAVA )
/* Generated */    #undef   TA_PREFIX
/* Generated */    #define  TA_PREFIX(x) TA_S_##x
/* Generated */ #endif
/* Generated */ #undef   INPUT_TYPE
/* Generated */ #define  INPUT_TYPE float
/* Generated */ #if defined( _MANAGED )
/* Generated */ enum class Core::RetCode Core::Mama( int    startIdx,
/* Generated */                                      int    endIdx,
/* Generated */                                      cli::array<float>^ inReal,
/* Generated */                                      double        optInFastLimit, /* From 0.01 to 0.99 */
/* Generated */                                      double        optInSlowLimit, /* From 0.01 to 0.99 */
/* Generated */                                      [Out]int%    outBegIdx,
/* Generated */                                      [Out]int%    outNBElement,
/* Generated */                                      cli::array<double>^  outMAMA,
/* Generated */                                      cli::array<double>^  outFAMA )
/* Generated */ #elif defined( _JAVA )
/* Generated */ public RetCode mama( int    startIdx,
/* Generated */                      int    endIdx,
/* Generated */                      float        inReal[],
/* Generated */                      double        optInFastLimit, /* From 0.01 to 0.99 */
/* Generated */                      double        optInSlowLimit, /* From 0.01 to 0.99 */
/* Generated */                      MInteger     outBegIdx,
/* Generated */                      MInteger     outNBElement,
/* Generated */                      double        outMAMA[],
/* Generated */                      double        outFAMA[] )
/* Generated */ #else
/* Generated */ TA_RetCode TA_S_MAMA( int    startIdx,
/* Generated */                       int    endIdx,
/* Generated */                       const float  inReal[],
/* Generated */                       double        optInFastLimit, /* From 0.01 to 0.99 */
/* Generated */                       double        optInSlowLimit, /* From 0.01 to 0.99 */
/* Generated */                       int          *outBegIdx,
/* Generated */                       int          *outNBElement,
/* Generated */                       double        outMAMA[],
/* Generated */                       double        outFAMA[] )
/* Generated */ #endif
/* Generated */ {
/* Generated */    int outIdx, i;
/* Generated */    int lookbackTotal, today;
/* Generated */    double tempReal, tempReal2;
/* Generated */    double adjustedPrevPeriod, period;
/* Generated */    int trailingWMAIdx;
/* Generated */    double periodWMASum, periodWMASub, trailingWMAValue;
/* Generated */    double smoothedValue;
/* Generated */    CONSTANT_DOUBLE(a) = 0.0962;
/* Generated */    CONSTANT_DOUBLE(b) = 0.5769;
/* Generated */    double hilbertTempReal;
/* Generated */    int hilbertIdx;
/* Generated */    HILBERT_VARIABLES( detrender );
/* Generated */    HILBERT_VARIABLES( Q1 );
/* Generated */    HILBERT_VARIABLES( jI );
/* Generated */    HILBERT_VARIABLES( jQ );
/* Generated */    double Q2, I2, prevQ2, prevI2, Re, Im;
/* Generated */    double I1ForOddPrev2,  I1ForOddPrev3;
/* Generated */    double I1ForEvenPrev2, I1ForEvenPrev3;
/* Generated */    double rad2Deg;
/* Generated */    double mama,fama,todayValue,prevPhase;
/* Generated */  #ifndef TA_FUNC_NO_RANGE_CHECK
/* Generated */     if( startIdx < 0 )
/* Generated */        return ENUM_VALUE(RetCode,TA_OUT_OF_RANGE_START_INDEX,OutOfRangeStartIndex);
/* Generated */     if( (endIdx < 0) || (endIdx < startIdx))
/* Generated */        return ENUM_VALUE(RetCode,TA_OUT_OF_RANGE_END_INDEX,OutOfRangeEndIndex);
/* Generated */     #if !defined(_JAVA)
/* Generated */     if( !inReal ) return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #endif 
/* Generated */     if( optInFastLimit == TA_REAL_DEFAULT )
/* Generated */        optInFastLimit = 5.000000e-1;
/* Generated */     else if( (optInFastLimit < 1.000000e-2) ||  (optInFastLimit > 9.900000e-1) )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     if( optInSlowLimit == TA_REAL_DEFAULT )
/* Generated */        optInSlowLimit = 5.000000e-2;
/* Generated */     else if( (optInSlowLimit < 1.000000e-2) ||  (optInSlowLimit > 9.900000e-1) )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #if !defined(_JAVA)
/* Generated */     if( !outMAMA )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     if( !outFAMA )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #endif 
/* Generated */  #endif 
/* Generated */    rad2Deg = 180.0 / (4.0 * std_atan(1));
/* Generated */    lookbackTotal = 32 + TA_GLOBALS_UNSTABLE_PERIOD(TA_FUNC_UNST_MAMA,Mama);
/* Generated */    if( startIdx < lookbackTotal )
/* Generated */       startIdx = lookbackTotal;
/* Generated */    if( startIdx > endIdx )
/* Generated */    {
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
/* Generated */       return ENUM_VALUE(RetCode,TA_SUCCESS,Success);
/* Generated */    }
/* Generated */    VALUE_HANDLE_DEREF(outBegIdx) = startIdx;
/* Generated */    trailingWMAIdx = startIdx - lookbackTotal;
/* Generated */    today = trailingWMAIdx;
/* Generated */    tempReal = inReal[today++];
/* Generated */    periodWMASub = tempReal;
/* Generated */    periodWMASum = tempReal;
/* Generated */    tempReal = inReal[today++];
/* Generated */    periodWMASub += tempReal;
/* Generated */    periodWMASum += tempReal*2.0;
/* Generated */    tempReal = inReal[today++];
/* Generated */    periodWMASub += tempReal;
/* Generated */    periodWMASum += tempReal*3.0;
/* Generated */    trailingWMAValue = 0.0;
/* Generated */    #define DO_PRICE_WMA(varNewPrice,varToStoreSmoothedValue) { \
/* Generated */       periodWMASub     += varNewPrice; \
/* Generated */       periodWMASub     -= trailingWMAValue; \
/* Generated */       periodWMASum     += varNewPrice*4.0; \
/* Generated */       trailingWMAValue  = inReal[trailingWMAIdx++]; \
/* Generated */       varToStoreSmoothedValue = periodWMASum*0.1; \
/* Generated */       periodWMASum -= periodWMASub; \
/* Generated */    }
/* Generated */    i = 9;
/* Generated */    do
/* Generated */    {
/* Generated */       tempReal = inReal[today++];
/* Generated */       DO_PRICE_WMA(tempReal,smoothedValue);
/* Generated */    } while( --i != 0);
/* Generated */    hilbertIdx = 0;
/* Generated */    INIT_HILBERT_VARIABLES(detrender);
/* Generated */    INIT_HILBERT_VARIABLES(Q1);
/* Generated */    INIT_HILBERT_VARIABLES(jI);
/* Generated */    INIT_HILBERT_VARIABLES(jQ);
/* Generated */    period = 0.0;
/* Generated */    outIdx = 0;
/* Generated */    prevI2 = prevQ2 = 0.0;
/* Generated */    Re     = Im     = 0.0;
/* Generated */    mama   = fama   = 0.0;
/* Generated */    I1ForOddPrev3 = I1ForEvenPrev3 = 0.0;
/* Generated */    I1ForOddPrev2 = I1ForEvenPrev2 = 0.0;
/* Generated */    prevPhase  = 0.0;
/* Generated */    while( today <= endIdx )
/* Generated */    {
/* Generated */       adjustedPrevPeriod = (0.075*period)+0.54;
/* Generated */       todayValue = inReal[today];
/* Generated */       DO_PRICE_WMA(todayValue,smoothedValue);
/* Generated */       if( (today%2) == 0 )
/* Generated */       {
/* Generated */          DO_HILBERT_EVEN(detrender,smoothedValue);
/* Generated */          DO_HILBERT_EVEN(Q1,detrender);
/* Generated */          DO_HILBERT_EVEN(jI,I1ForEvenPrev3);
/* Generated */          DO_HILBERT_EVEN(jQ,Q1);
/* Generated */          if( ++hilbertIdx == 3 )
/* Generated */             hilbertIdx = 0;
/* Generated */          Q2 = (0.2*(Q1 + jI)) + (0.8*prevQ2);
/* Generated */          I2 = (0.2*(I1ForEvenPrev3 - jQ)) + (0.8*prevI2);
/* Generated */          I1ForOddPrev3 = I1ForOddPrev2;
/* Generated */          I1ForOddPrev2 = detrender;
/* Generated */          if( I1ForEvenPrev3 != 0.0 )
/* Generated */             tempReal2 = (std_atan(Q1/I1ForEvenPrev3)*rad2Deg);
/* Generated */          else
/* Generated */             tempReal2 = 0.0;
/* Generated */       }
/* Generated */       else
/* Generated */       {
/* Generated */          DO_HILBERT_ODD(detrender,smoothedValue);
/* Generated */          DO_HILBERT_ODD(Q1,detrender);
/* Generated */          DO_HILBERT_ODD(jI,I1ForOddPrev3);
/* Generated */          DO_HILBERT_ODD(jQ,Q1);
/* Generated */          Q2 = (0.2*(Q1 + jI)) + (0.8*prevQ2);
/* Generated */          I2 = (0.2*(I1ForOddPrev3 - jQ)) + (0.8*prevI2);
/* Generated */          I1ForEvenPrev3 = I1ForEvenPrev2;
/* Generated */          I1ForEvenPrev2 = detrender;
/* Generated */          if( I1ForOddPrev3 != 0.0 )
/* Generated */             tempReal2 = (std_atan(Q1/I1ForOddPrev3)*rad2Deg);
/* Generated */          else
/* Generated */             tempReal2 = 0.0;
/* Generated */       }
/* Generated */       tempReal  = prevPhase - tempReal2;
/* Generated */       prevPhase = tempReal2;
/* Generated */       if( tempReal < 1.0 )
/* Generated */          tempReal = 1.0;
/* Generated */       if( tempReal > 1.0 )
/* Generated */       {
/* Generated */          tempReal = optInFastLimit/tempReal;
/* Generated */          if( tempReal < optInSlowLimit )
/* Generated */             tempReal = optInSlowLimit;
/* Generated */       }
/* Generated */       else
/* Generated */       {
/* Generated */          tempReal = optInFastLimit;
/* Generated */       }  
/* Generated */       mama = (tempReal*todayValue)+((1-tempReal)*mama);
/* Generated */       tempReal *= 0.5;
/* Generated */       fama = (tempReal*mama)+((1-tempReal)*fama);
/* Generated */       if( today >= startIdx )
/* Generated */       {
/* Generated */          outMAMA[outIdx] = mama;
/* Generated */          outFAMA[outIdx++] = fama;
/* Generated */       }
/* Generated */       Re = (0.2*((I2*prevI2)+(Q2*prevQ2)))+(0.8*Re);
/* Generated */       Im = (0.2*((I2*prevQ2)-(Q2*prevI2)))+(0.8*Im);
/* Generated */       prevQ2 = Q2;
/* Generated */       prevI2 = I2;
/* Generated */       tempReal = period;
/* Generated */       if( (Im != 0.0) && (Re != 0.0) )
/* Generated */          period = 360.0 / (std_atan(Im/Re)*rad2Deg);
/* Generated */       tempReal2 = 1.5*tempReal;
/* Generated */       if( period > tempReal2)
/* Generated */          period = tempReal2;
/* Generated */       tempReal2 = 0.67*tempReal;
/* Generated */       if( period < tempReal2 )
/* Generated */          period = tempReal2;
/* Generated */       if( period < 6 )
/* Generated */          period = 6;
/* Generated */       else if( period > 50 )
/* Generated */          period = 50;
/* Generated */       period = (0.2*period) + (0.8 * tempReal);
/* Generated */       today++;
/* Generated */    }
/* Generated */    VALUE_HANDLE_DEREF(outNBElement) = outIdx;
/* Generated */    return ENUM_VALUE(RetCode,TA_SUCCESS,Success);
/* Generated */ }
/* Generated */ 
/* Generated */ #if defined( _MANAGED )
/* Generated */ }}} // Close namespace TicTacTec.TA.Lib
/* Generated */ #endif
/**** END GENCODE SECTION 5 - DO NOT DELETE THIS LINE ****/

