/*
Copyright 2014, Jernej Kovacic

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/**
 * @file
 *
 * Definitions of mappings of all supported system clock
 * frequencies to their corresponding divisors.
 *
 * Note that the integer and fractional parts of a frequency
 * in a define are separated by an underscore sign ('_').
 * Fractional parts are rounded to 3 decimal places.
 *
 * For more details, see Table 5-6 at page 224 of
 * Tiva(TM) TM4C123GH6PM Microcontroller Data Sheet,
 * available at:
 * http://www.ti.com/lit/ds/symlink/tm4c123gh6pm.pdf
 *
 * @author Jernej Kovacic
 */


#ifndef _PLL_FREQ_DIVISORS_H_
#define _PLL_FREQ_DIVISORS_H_


#define DIV_FREQ_80_MHZ                  ( 5 )
#define DIV_FREQ_66_667_MHZ              ( 6 )
#define DIV_FREQ_50_MHZ                  ( 8 )
#define DIV_FREQ_44_444_MHZ              ( 9 )
#define DIV_FREQ_40_MHZ                  ( 10 )
#define DIV_FREQ_36_364_MHZ              ( 11 )
#define DIV_FREQ_33_333_MHZ              ( 12 )
#define DIV_FREQ_30_769_MHZ              ( 13 )
#define DIV_FREQ_28_571_MHZ              ( 14 )
#define DIV_FREQ_26_667_MHZ              ( 15 )
#define DIV_FREQ_25_MHZ                  ( 16 )
#define DIV_FREQ_23_594_MHZ              ( 17 )
#define DIV_FREQ_22_222_MHZ              ( 18 )
#define DIV_FREQ_21_053_MHZ              ( 19 )
#define DIV_FREQ_20_MHZ                  ( 20 )
#define DIV_FREQ_19_048_MHZ              ( 21 )
#define DIV_FREQ_18_182_MHZ              ( 22 )
#define DIV_FREQ_17_391_MHZ              ( 23 )
#define DIV_FREQ_16_667_MHZ              ( 24 )
#define DIV_FREQ_16_MHZ                  ( 25 )
#define DIV_FREQ_15_385_MHZ              ( 26 )
#define DIV_FREQ_14_815_MHZ              ( 27 )
#define DIV_FREQ_14_286_MHZ              ( 28 )
#define DIV_FREQ_13_793_MHZ              ( 29 )
#define DIV_FREQ_13_333_MHZ              ( 30 )
#define DIV_FREQ_12_903_MHZ              ( 31 )
#define DIV_FREQ_12_5_MHZ                ( 32 )
#define DIV_FREQ_12_121_MHZ              ( 33 )
#define DIV_FREQ_11_765_MHZ              ( 34 )
#define DIV_FREQ_11_429_MHZ              ( 35 )
#define DIV_FREQ_11_111_MHZ              ( 36 )
#define DIV_FREQ_10_811_MHZ              ( 37 )
#define DIV_FREQ_10_526_MHZ              ( 38 )
#define DIV_FREQ_10_256_MHZ              ( 39 )
#define DIV_FREQ_10_MHZ                  ( 40 )
#define DIV_FREQ_9_756_MHZ               ( 41 )
#define DIV_FREQ_9_524_MHZ               ( 42 )
#define DIV_FREQ_9_302_MHZ               ( 43 )
#define DIV_FREQ_9_091_MHZ               ( 44 )
#define DIV_FREQ_8_889_MHZ               ( 45 )
#define DIV_FREQ_8_696_MHZ               ( 46 )
#define DIV_FREQ_8_511_MHZ               ( 47 )
#define DIV_FREQ_8_333_MHZ               ( 48 )
#define DIV_FREQ_8_163_MHZ               ( 49 )
#define DIV_FREQ_8_MHZ                   ( 50 )
#define DIV_FREQ_7_843_MHZ               ( 51 )
#define DIV_FREQ_7_692_MHZ               ( 52 )
#define DIV_FREQ_7_547_MHZ               ( 53 )
#define DIV_FREQ_7_407_MHZ               ( 54 )
#define DIV_FREQ_7_273_MHZ               ( 55 )
#define DIV_FREQ_7_143_MHZ               ( 56 )
#define DIV_FREQ_7_018_MHZ               ( 57 )
#define DIV_FREQ_6_897_MHZ               ( 58 )
#define DIV_FREQ_6_780_MHZ               ( 59 )
#define DIV_FREQ_6_667_MHZ               ( 60 )
#define DIV_FREQ_6_557_MHZ               ( 61 )
#define DIV_FREQ_6_452_MHZ               ( 62 )
#define DIV_FREQ_6_349_MHZ               ( 63 )
#define DIV_FREQ_6_25_MHZ                ( 64 )
#define DIV_FREQ_6_154_MHZ               ( 65 )
#define DIV_FREQ_6_061_MHZ               ( 66 )
#define DIV_FREQ_5_970_MHZ               ( 67 )
#define DIV_FREQ_5_882_MHZ               ( 68 )
#define DIV_FREQ_5_797_MHZ               ( 69 )
#define DIV_FREQ_5_714_MHZ               ( 70 )
#define DIV_FREQ_5_634_MHZ               ( 71 )
#define DIV_FREQ_5_556_MHZ               ( 72 )
#define DIV_FREQ_5_480_MHZ               ( 73 )
#define DIV_FREQ_5_405_MHZ               ( 74 )
#define DIV_FREQ_5_333_MHZ               ( 75 )
#define DIV_FREQ_5_263_MHZ               ( 76 )
#define DIV_FREQ_5_195_MHZ               ( 77 )
#define DIV_FREQ_5_128_MHZ               ( 78 )
#define DIV_FREQ_5_063_MHZ               ( 79 )
#define DIV_FREQ_5_MHZ                   ( 80 )
#define DIV_FREQ_4_938_MHZ               ( 81 )
#define DIV_FREQ_4_878_MHZ               ( 82 )
#define DIV_FREQ_4_819_MHZ               ( 83 )
#define DIV_FREQ_4_762_MHZ               ( 84 )
#define DIV_FREQ_4_706_MHZ               ( 85 )
#define DIV_FREQ_4_651_MHZ               ( 86 )
#define DIV_FREQ_4_598_MHZ               ( 87 )
#define DIV_FREQ_4_546_MHZ               ( 88 )
#define DIV_FREQ_4_494_MHZ               ( 89 )
#define DIV_FREQ_4_444_MHZ               ( 90 )
#define DIV_FREQ_4_396_MHZ               ( 91 )
#define DIV_FREQ_4_348_MHZ               ( 92 )
#define DIV_FREQ_4_301_MHZ               ( 93 )
#define DIV_FREQ_4_255_MHZ               ( 94 )
#define DIV_FREQ_4_211_MHZ               ( 95 )
#define DIV_FREQ_4_167_MHZ               ( 96 )
#define DIV_FREQ_4_124_MHZ               ( 97 )
#define DIV_FREQ_4_082_MHZ               ( 98 )
#define DIV_FREQ_4_040_MHZ               ( 99 )
#define DIV_FREQ_4_MHZ                   ( 100 )
#define DIV_FREQ_3_960_MHZ               ( 101 )
#define DIV_FREQ_3_922_MHZ               ( 102 )
#define DIV_FREQ_3_884_MHZ               ( 103 )
#define DIV_FREQ_3_846_MHZ               ( 104 )
#define DIV_FREQ_3_810_MHZ               ( 105 )
#define DIV_FREQ_3_774_MHZ               ( 106 )
#define DIV_FREQ_3_738_MHZ               ( 107 )
#define DIV_FREQ_3_704_MHZ               ( 108 )
#define DIV_FREQ_3_670_MHZ               ( 109 )
#define DIV_FREQ_3_636_MHZ               ( 110 )
#define DIV_FREQ_3_604_MHZ               ( 111 )
#define DIV_FREQ_3_571_MHZ               ( 112 )
#define DIV_FREQ_3_540_MHZ               ( 113 )
#define DIV_FREQ_3_509_MHZ               ( 114 )
#define DIV_FREQ_3_478_MHZ               ( 115 )
#define DIV_FREQ_3_448_MHZ               ( 116 )
#define DIV_FREQ_3_419_MHZ               ( 117 )
#define DIV_FREQ_3_390_MHZ               ( 118 )
#define DIV_FREQ_3_361_MHZ               ( 119 )
#define DIV_FREQ_3_333_MHZ               ( 120 )
#define DIV_FREQ_3_306_MHZ               ( 121 )
#define DIV_FREQ_3_279_MHZ               ( 122 )
#define DIV_FREQ_3_252_MHZ               ( 123 )
#define DIV_FREQ_3_226_MHZ               ( 124 )
#define DIV_FREQ_3_2_MHZ                 ( 125 )
#define DIV_FREQ_3_175_MHZ               ( 126 )
#define DIV_FREQ_3_150_MHZ               ( 127 )
#define DIV_FREQ_3_125_MHZ               ( 128 )


#endif  /* _PLL_FREQ_DIVISORS_H_ */
