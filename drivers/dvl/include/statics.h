
private: // Static data about the DVL binary protocol

#define EMPTY 0
// Help Commands
static constexpr Command cGeneralCommandHelp             = {"?",                                   "cGeneralCommandHelp"};
static constexpr Command cBottomCommandHelp              = {"B?",                                  "cBottomCommandHelp"};
static constexpr Command cControlCommandHelp             = {"C?",                                  "cControlCommandHelp"};
static constexpr Command cEnvironmentCommandHelp         = {"E?",                                  "cEnvironmentalCommandHelp"};
static constexpr Command cLoopRecorderCommandHelp        = {"M?",                                  "cLoopRecorderCommandHelp"};
static constexpr Command cPerformanceTestCommandHelp     = {"P?",                                  "cPerformanceTestHelp"};
static constexpr Command cSensorCommandHelp              = {"S?",                                  "cSensorCommandHelp"};
static constexpr Command cTimeCommandHelp                = {"T?",                                  "cTimeCommandHelp"};
static constexpr Command cWaterProfilingCommandHelp      = {"W?",                                  "cWaterProfilingHelp"};
// Bottom Tracking Commands
static constexpr Command cEvaluationAmplitudeMinimum     = {"#BA%3d",                              "cEvaluationAmplitudeMinimum"};
static constexpr Command cBottomBlankingInterval         = {"#BB%4d",                              "cBottomBlankingInterval"};
static constexpr Command cBottomCorrelationMagnitude     = {"#BC%3d",                              "cBottomCorrelationMagnitude"};
static constexpr Command cBottomErrorVelocityMaximum     = {"#BE%4f",                              "cBottomErrorVelocityMaximum"};
static constexpr Command cBottomDepthGuess               = {"#BF%5d",                              "cBottomDepthGuess"};
static constexpr Command cBottomGainSwitchDepth          = {"#BI%3d",                              "cBottomGainSwitchDepth"};
static constexpr Command cBottomDataOut                  = {"#BJ %1d%1d0 %1d%1d%1d 000",           "cBottomDataOut"}; // BINARY
static constexpr Command cWaterMassPing                  = {"#BK%1d",                              "cWaterMassPing"};
static constexpr Command cWaterMassParameters            = {"#BL%3d,%4d,%4d",                      "cWaterMassParameters"};
static constexpr Command cSpeedLogControl                = {"#BN%1d,%3d",                          "cSpeedLogControl"};
static constexpr Command cDistanceMeasureFilterConstant  = {"#BO%3d",                              "cDistanceMeasureFilterConstant"};
static constexpr Command cBottomTrackPingsPerEnsemble    = {"BP%3d",                               "cBottomTrackPingsPerEnsemble"};
static constexpr Command cClearDistanceTravelled         = {"#BS",                                 "cClearDistanceTravelled"};
static constexpr Command cMaximumTrackingDepth           = {"BX%5d",                               "cMaximumTrackingDepth"};
// Control Commands
static constexpr Command cSerialPortControl              = {"CB%1d%1d%1d",                         "cSerialPortControl"};
static constexpr Command cFlowControl                    = {"CF%1d%1d%1d%1d0",                     "cFlowControl"}; // BINARY
static constexpr Command cKeepParameters                 = {"CK",                                  "cKeepParameters"};
static constexpr Command cRetrieveUserParameters         = {"CR0",                                 "cRetrieveUserParameters"};
static constexpr Command cRetrieveFactoryParameters      = {"CR1",                                 "cRetrieveFactoryParameters"};
static constexpr Command cStartPinging                   = {"CS",                                  "cStartPinging"};
static constexpr Command cTurnkeyDisable                 = {"#CT0",                                "cTurnkeyDisable"};
static constexpr Command cTurnkeyEnable                  = {"#CT1",                                "cTurnkeyEnable"};
static constexpr Command cSetOutTrigger                  = {"#CO %1d %1d",                         "cSetOutTriger"};
static constexpr Command cSetInputTrigger                = {"CX %1d %1d %5d",                      "cSetInputTrigger"};
// Environment Commands
static constexpr Command cStaticHeadingOffset            = {"EA%+5d",                              "cHeadingAlignment"};
static constexpr Command cManualSpeedOfSound             = {"#EC%4d",                              "cManualSpeedOfSound"};
static constexpr Command cLiveDepth                      = {"ED%5d",                               "cLiveDepth"};
static constexpr Command cEnvironmentalDataOutput        = {"#EE 0000%1d%1d%1d",                   "cEnvironmentalDataOutput"}; // BINARY
static constexpr Command cLiveHeading                    = {"#EH%5d,%1d",                          "cLiveHeading"};
static constexpr Command cStaticRollOffset               = {"#EI%+5d",                             "cStaticRollOffset"};
static constexpr Command cStaticPitchOffset              = {"#EJ%+5d",                             "cStaticPitchOffset"};
static constexpr Command cLivePitchAndRoll               = {"#EP%+5d,%+5d,%1d",                    "cLivePitchAndRoll"};
static constexpr Command cLiveSalinity                   = {"ES%2d",                               "cLiveSalinity"};
static constexpr Command cLiveTemperature                = {"#ET%+4d",                             "cLiveTemperature"};
static constexpr Command cOrientationResolution          = {"#EU%1d",                              "cOrientationResolution"};
static constexpr Command cStaticMagneticHeadingOffset    = {"#EV%+5d",                             "cStaticMagneticHeadingOffset"};
static constexpr Command cCoordinateTransformation       = {"EX%2d%1d%1d%1d",                      "cCoordinateTransformation"}; // BINARY
static constexpr Command cDopplerParameterSource         = {"#EY %1d %1d %1d %1d %1d %1d %1d %1d", "cDopplerParameterSource"}; // BINARY
static constexpr Command cSensorSource                   = {"EZ%1d%1d%1d%1d%1d%1d%1d%1d",          "SensorSource"};
// Loop Recorder Commands
static constexpr Command cRecorderErase                  = {"ME ErAsE",                            "cRecorderErase"};
static constexpr Command cShowMemoryUsage                = {"MM",                                  "cShowMemoryUsage"};
static constexpr Command cSetFileName                    = {"MN %s",                               "cSetFileName"};
static constexpr Command cRecorderDisable                = {"MR0",                                 "cRecorderDisable"};
static constexpr Command cRecorderEnable                 = {"MR1",                                 "cRecorderEnable"};
static constexpr Command cFileDownload                   = {"MY",                                  "cFileDownload"};
// Performance Commands
static constexpr Command cPreDeploymentTests             = {"PA",                                  "cPreDeploymentTests"};
static constexpr Command cInteractiveTest                = {"PC2",                                 "cInteractiveTest"};
static constexpr Command cSendRawWaterCurrentData        = {"#PD0",                                "cSendRawWaterCurrentData"};
static constexpr Command cSendWithoutSensorData          = {"#PD4",                                "cSendWithoutSensorData"};
static constexpr Command cSendWithSensorData             = {"#PD5",                                "cSendWithSensorData"};
static constexpr Command cSendASCIIData                  = {"#PD6",                                "cSendASCIIData"};
static constexpr Command cDisplaySystemConfiguration     = {"PS0",                                 "cDisplaySystemConfiguration"};
static constexpr Command cDisplayFixedLeader             = {"PS1",                                 "cDisplayFixedLeader"};
static constexpr Command cDisplayTransformMatrix         = {"PS3",                                 "cDisplayTransformMatrix"};
static constexpr Command cDisplayPingSequence            = {"PS4",                                 "cDisplayPingSequence"};
static constexpr Command cTestDVLRecievePath             = {"PT3",                                 "cTestDVLRecievePath"};
static constexpr Command cTestTxRxLoop                   = {"PT5",                                 "cTestTxRxLoop"};
static constexpr Command cTestWiring                     = {"PT7",                                 "cTestWiring"};
static constexpr Command cTestDVLRecievePathRepeat       = {"PT103",                               "cTestDVLRecievePathRepeat"};
static constexpr Command cTestTxRxLoopRepeat             = {"PT105",                               "cTestTxRxLoopRepeat"};
static constexpr Command cTestWiringRepeat               = {"PT107",                               "cTestWiringRepeat"};
static constexpr Command cTestAll                        = {"PT200",                               "cTestAll"};
static constexpr Command cTestAllRepeat                  = {"PT300",                               "cTestAllRepeat"};
// Sensor Commands
static constexpr Command cSendSensorCommand              = {"SC %2d %1d %5d \"%s\"",               "cSendSensorCommand"};
static constexpr Command cSensorDataOut                  = {"SD %02d %1d%1d%1d%1d00000",            "cSensorDataOut"}; //BINARY
static constexpr Command cAuxSensorAuxMenu               = {"SM",                                  "cAuxSensorAuxMenu"};
static constexpr Command cPressureSensorOffset           = {"#SO %3.3f",                           "cPressureSensorOffset"};
static constexpr Command cSensorPortAssignment           = {"SP%1d %2d",                           "cSensorPortAssignment"};
static constexpr Command cSensorReset                    = {"SR %2d",                              "cSensorReset"};
// Time Commands
static constexpr Command cTimePerEnsemble                = {"TE %02d:%02d:%05.2f",                     "cTimePerEnsemble"};
static constexpr Command cTimeBetweenPings               = {"TP %02d:%05.2f",                         "cTimeBetweenPings"};
static constexpr Command cSetRealTimeClock               = {"TS %04d/%02d/%02d,%02d:%02d:%02.2f",         "cSetRealTimeClock"}; //Y2K-Compliant
// Water Profiling Commands
static constexpr Command cFalseTargetThreshold           = {"WA%3d",                               "cFalseTargetThreshold"};
static constexpr Command cBandwidthMode                  = {"WB%1d",                               "cBandwithMode"};
static constexpr Command cCorrelationThreshold           = {"#WC%3d",                              "cCorrelationThreshold"};
static constexpr Command cDataOut                        = {"WD %1d%1d%1d %1d%1d0 000",     "cDataOut"}; // BINARY
static constexpr Command cErrorVelocityThreshold         = {"#WE%4d",                              "cErrorVelocityThreshold"};
static constexpr Command cBlankingDistance               = {"WF%3d",                               "cBlankingDistance"};
static constexpr Command cReceiverGain                   = {"#WJ%1d",                              "cReceiverGain"};
static constexpr Command cNumberOfBins                   = {"WN%3d",                               "cNumberOfBins"};
static constexpr Command cPingsPerEnsemble               = {"WP%5d",                               "cPingsPerEnsemble"};
static constexpr Command cBinSize                        = {"WS%4d",                               "cBinSize"};
static constexpr Command cTransmitLength                 = {"#WT%4d",                              "cTransmitLength"};
static constexpr Command cAmbiguityVelocity              = {"#WV%3d",                              "cAmbiguityVelocity"};
// Feature Commands
static constexpr Command cListFeatures                   = {"OL",                                  "cListFeatures"};
static constexpr Command cInstallFeature                 = {"OI",                                  "cInstallFeature"};

static constexpr frameid_t kPD0HeaderID                  = 0x7f7f;
static constexpr frameid_t kPD0FixedLeaderID             = 0x0000;
static constexpr frameid_t kPD0VariableLeaderID          = 0x0080;
static constexpr frameid_t kPD0VelocityDataID            = 0x0100;
static constexpr frameid_t kPD0CorrelationMagnitudeID    = 0x0200;
static constexpr frameid_t kPD0EchoIntensityID           = 0x0300;
static constexpr frameid_t kPD0PercentGoodID             = 0x0400;
static constexpr frameid_t kPD0StatusDataID              = 0x0500;
static constexpr frameid_t kPD0BottomTrackID             = 0x0600;
static constexpr frameid_t kPD0EnvironmentID             = 0x3000;
static constexpr frameid_t kPD0BottomTrackCommandID      = 0x5800;
static constexpr frameid_t kPD0BottomTrackHighResID      = 0x5803;
static constexpr frameid_t kPD0BottomTrackRangeID        = 0x5804;
static constexpr frameid_t kPD0SensorDataID              = 0x3001;
static constexpr frameid_t kPD4HeaderID                  = 0x007d;
static constexpr frameid_t kPD5HeaderID                  = 0x017d;

public:
static constexpr DVLSpeed k300                           = {0, B300};
static constexpr DVLSpeed k1200                          = {1, B1200};
static constexpr DVLSpeed k2400                          = {2, B2400};
static constexpr DVLSpeed k4800                          = {3, B4800};
static constexpr DVLSpeed k9600                          = {4, B9600};
static constexpr DVLSpeed k19200                         = {5, B19200};
static constexpr DVLSpeed k38400                         = {6, B38400};
static constexpr DVLSpeed k57600                         = {7, B57600};
static constexpr DVLSpeed k115200                        = {8, B115200};

























