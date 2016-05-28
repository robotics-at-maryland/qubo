
private:
#define EMPTY 0
/*****************************************************************************************************************
 * Below is all hardcoded data from the protocol spec'd for the TRAX.
 *                     | Command Name                  | ID  | Payload Size             | Command String
 *****************************************************************************************************************/
static constexpr Command kGetModInfo                  = {0x01, EMPTY,                     "kGetModInfo"};
static constexpr Command kGetModInfoResp              = {0x02, sizeof(ModInfo),           "kGetModInfoResp"};
static constexpr Command kSetDataComponents           = {0x03, sizeof(RawDataFields),     "kSetDataComponents"};
static constexpr Command kGetData                     = {0x04, EMPTY,                     "kGetData"};
static constexpr Command kGetDataResp                 = {0x05, sizeof(RawData),           "kGetDataResp"};
static constexpr Command kSetConfigBoolean            = {0x06, sizeof(ConfigBoolean),     "kSetConfigBoolean"};
static constexpr Command kSetConfigFloat32            = {0x06, sizeof(ConfigFloat32),     "kSetConfigFloat32"};
static constexpr Command kSetConfigUInt8              = {0x06, sizeof(ConfigUInt8),       "kSetConfigUInt8"};
static constexpr Command kSetConfigUInt32             = {0x06, sizeof(ConfigUInt32),      "kSetConfigUInt32"};
static constexpr Command kGetConfig                   = {0x07, sizeof(config_id_t),       "kGetConfig"};
static constexpr Command kGetConfigRespBoolean        = {0x08, sizeof(ConfigBoolean),     "kGetConfigRespBoolean"};
static constexpr Command kGetConfigRespFloat32        = {0x08, sizeof(ConfigFloat32),     "kGetConfigRespFloat32"};
static constexpr Command kGetConfigRespUInt8          = {0x08, sizeof(ConfigUInt8),       "kGetConfigRespUInt8"};
static constexpr Command kGetConfigRespUInt32         = {0x08, sizeof(ConfigUInt32),      "kGetConfigRespUInt32"};
static constexpr Command kSave                        = {0x09, EMPTY,                     "kSave"};
static constexpr Command kStartCal                    = {0x0a, sizeof(CalOption),         "kStartCal"};
static constexpr Command kStopCal                     = {0x0b, EMPTY,                     "kStopCal"};
static constexpr Command kSetFIRFiltersZero           = {0x0c, F_0*sizeof(double)+3,      "kSetFIRFiltersZero"};
static constexpr Command kSetFIRFiltersFour           = {0x0c, F_4*sizeof(double)+3,      "kSetFIRFiltersFour"};
static constexpr Command kSetFIRFiltersEight          = {0x0c, F_8*sizeof(double)+3,      "kSetFIRFiltersEight"};
static constexpr Command kSetFIRFiltersSixteen        = {0x0c, F_16*sizeof(double)+3,     "kSetFIRFiltersSixteen"};
static constexpr Command kSetFIRFiltersThirtyTwo      = {0x0c, F_32*sizeof(double)+3,     "kSetFIRFiltersThirtyTwo"};
static constexpr Command kGetFIRFilters               = {0x0d, sizeof(FIRFilter),         "kGetFIRFilters"};
static constexpr Command kGetFIRFiltersRespZero       = {0x0e, F_0*sizeof(double)+3,      "kGetFIRFiltersRespZero"};
static constexpr Command kGetFIRFiltersRespFour       = {0x0e, F_4*sizeof(double)+3,      "kGetFIRFiltersRespFour"};
static constexpr Command kGetFIRFiltersRespEight      = {0x0e, F_8*sizeof(double)+3,      "kGetFIRFiltersRespEight"};
static constexpr Command kGetFIRFiltersRespSixteen    = {0x0e, F_16*sizeof(double)+3,     "kGetFIRFiltersRespSixteen"};
static constexpr Command kGetFIRFiltersRespThirtyTwo  = {0x0e, F_32*sizeof(double)+3,     "kGetFIRFiltersRespThirtyTwo"};
static constexpr Command kPowerDown                   = {0x0f, EMPTY,                     "kPowerDown"};
static constexpr Command kSaveDone                    = {0x10, sizeof(SaveError),         "kSaveDone"};
static constexpr Command kUserCalSampleCount          = {0x11, sizeof(SampleCount),       "kUserCalSampleCount"};
static constexpr Command kUserCalScore                = {0x12, sizeof(UserCalScore),      "kUserCalScore"};
static constexpr Command kSetConfigDone               = {0x13, EMPTY,                     "kSetConfigDone"};
static constexpr Command kSetFIRFiltersDone           = {0x14, EMPTY,                     "kSetFIRFiltersDone"};
static constexpr Command kStartContinuousMode         = {0x15, EMPTY,                     "kStartContinuousMode"};
static constexpr Command kStopContinousMode           = {0x16, EMPTY,                     "kStopContinousMode"};
static constexpr Command kPowerUpDone                 = {0x17, EMPTY,                     "kPowerUpDone"};
static constexpr Command kSetAcqParams                = {0x18, sizeof(AcqParams),         "kSetAcqParams"};
static constexpr Command kGetAcqParams                = {0x19, EMPTY,                     "kGetAcqParams"};
static constexpr Command kSetAcqParamsDone            = {0x1a, EMPTY,                     "kSetAcqParamsDone"};
static constexpr Command kGetAcqParamsResp            = {0x1b, sizeof(AcqParams),         "kGetAcqParamsResp"};
static constexpr Command kPowerDownDone               = {0x1c, EMPTY,                     "kPowerDownDone"};
static constexpr Command kFactoryMagCoeff             = {0x1d, EMPTY,                     "kFactoryMagCoeff"};
static constexpr Command kFactoryMagCoeffDone         = {0x1e, EMPTY,                     "kFactoryMagCoeffDone"};
static constexpr Command kTakeUserCalSample           = {0x1f, EMPTY,                     "kTakeUserCalSample"};
static constexpr Command kFactoryAccelCoeff           = {0x24, EMPTY,                     "kFactoryAccelCoeff"};
static constexpr Command kFactoryAccelCoeffDone       = {0x25, EMPTY,                     "kFactoryAccelCoeffDone"};
static constexpr Command kSetFunctionalMode           = {0x4f, sizeof(FunctionalMode),    "kSetFunctionalMode"};
static constexpr Command kGetFunctionalMode           = {0x50, EMPTY,                     "kGetFunctionalMode"};
static constexpr Command kGetFunctionalModeResp       = {0x51, sizeof(FunctionalMode),    "kGetFunctionalModeResp"};
static constexpr Command kSetResetRef                 = {0x6e, EMPTY,                     "kSetResetRef"};
static constexpr Command kSetMagTruthMethod           = {0x77, sizeof(MagTruthMethod),    "kSetMagTruthMethod"};
static constexpr Command kGetMagTruthMethod           = {0x78, EMPTY,                     "kGetMagTruthMethod"};
static constexpr Command kGetMagTruthMethodResp       = {0x79, sizeof(MagTruthMethod),    "kGetMagTruthMethodResp"};
#undef EMPTY

static constexpr config_id_t kDeclination             = 1;
static constexpr config_id_t kTrueNorth               = 2;
static constexpr config_id_t kBigEndian               = 6;
static constexpr config_id_t kMountingRef             = 10;
static constexpr config_id_t kUserCalNumPoints        = 12;
static constexpr config_id_t kUserCalAutoSampling     = 13;
static constexpr config_id_t kBaudRate                = 14;
static constexpr config_id_t kMilOut                  = 15;
static constexpr config_id_t kHPRDuringCal            = 16;
static constexpr config_id_t kMagCoeffSet             = 18;
static constexpr config_id_t kAccelCoeffSet           = 19;

static constexpr data_id_t kPitch                     = 0x18;
static constexpr data_id_t kRoll                      = 0x19;
static constexpr data_id_t kHeadingStatus             = 0x4f;
static constexpr data_id_t kQuaternion                = 0x4d;
static constexpr data_id_t kTemperature               = 0x07;
static constexpr data_id_t kDistortion                = 0x08;
static constexpr data_id_t kCalStatus                 = 0x09;
static constexpr data_id_t kAccelX                    = 0x15;
static constexpr data_id_t kAccelY                    = 0x16;
static constexpr data_id_t kAccelZ                    = 0x17;
static constexpr data_id_t kMagX                      = 0x1b;
static constexpr data_id_t kMagY                      = 0x1c;
static constexpr data_id_t kMagZ                      = 0x1d;
static constexpr data_id_t kGyroX                     = 0x4a;
static constexpr data_id_t kGyroY                     = 0x4b;
static constexpr data_id_t kGyroZ                     = 0x4c;

/** static const struct with the permanent data type config. */
static constexpr RawDataFields dataConfig             = {
   10, kQuaternion, kGyroX, kGyroY, kGyroZ,
   kAccelX, kAccelY, kAccelZ, kMagX, kMagY, kMagZ};

public:
static constexpr AHRSSpeed k0        = {0,    B0};
static constexpr AHRSSpeed k2400     = {4,    B2400};
static constexpr AHRSSpeed k4800     = {6,    B4800};
static constexpr AHRSSpeed k9600     = {8,    B9600};
static constexpr AHRSSpeed k19200    = {10,   B19200};
static constexpr AHRSSpeed k38400    = {12,   B38400};
static constexpr AHRSSpeed k57600    = {13,   B57600};
static constexpr AHRSSpeed k115200   = {14,   B115200};
