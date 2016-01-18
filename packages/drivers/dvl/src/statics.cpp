
constexpr DVL::Command DVL::cGeneralCommandHelp;
constexpr DVL::Command DVL::cBottomCommandHelp;
constexpr DVL::Command DVL::cControlCommandHelp;
constexpr DVL::Command DVL::cEnvironmentCommandHelp;
constexpr DVL::Command DVL::cLoopRecorderCommandHelp;
constexpr DVL::Command DVL::cPerformanceTestCommandHelp;
constexpr DVL::Command DVL::cSensorCommandHelp;
constexpr DVL::Command DVL::cTimeCommandHelp;
constexpr DVL::Command DVL::cWaterProfilingCommandHelp;

constexpr DVL::Command DVL::cEvaluationAmplitudeMinimum;
constexpr DVL::Command DVL::cBottomBlankingInterval;
constexpr DVL::Command DVL::cBottomCorrelationMagnitude;
constexpr DVL::Command DVL::cBottomErrorVelocityMaximum;
constexpr DVL::Command DVL::cBottomDepthGuess;
constexpr DVL::Command DVL::cBottomGainSwitchDepth;
constexpr DVL::Command DVL::cBottomDataOut;
constexpr DVL::Command DVL::cWaterMassPing;
constexpr DVL::Command DVL::cWaterMassParameters;
constexpr DVL::Command DVL::cSpeedLogControl;
constexpr DVL::Command DVL::cDistanceMeasureFilterConstant;
constexpr DVL::Command DVL::cBottomTrackPingsPerEnsemble;
constexpr DVL::Command DVL::cClearDistanceTravelled;
constexpr DVL::Command DVL::cMaximumTrackingDepth;

constexpr DVL::Command DVL::cSerialPortControl;
constexpr DVL::Command DVL::cFlowControl;
constexpr DVL::Command DVL::cKeepParameters;
constexpr DVL::Command DVL::cRetrieveUserParameters;
constexpr DVL::Command DVL::cRetrieveFactoryParameters;
constexpr DVL::Command DVL::cStartPinging;
constexpr DVL::Command DVL::cTurnkeyDisable;
constexpr DVL::Command DVL::cTurnkeyEnable;
constexpr DVL::Command DVL::cSetOutTrigger;
constexpr DVL::Command DVL::cSetInputTrigger;

constexpr DVL::Command DVL::cStaticHeadingOffset;
constexpr DVL::Command DVL::cManualSpeedOfSound;
constexpr DVL::Command DVL::cLiveDepth;
constexpr DVL::Command DVL::cEnvironmentalDataOutput;
constexpr DVL::Command DVL::cLiveHeading;
constexpr DVL::Command DVL::cStaticRollOffset;
constexpr DVL::Command DVL::cStaticPitchOffset;
constexpr DVL::Command DVL::cLivePitchAndRoll;
constexpr DVL::Command DVL::cLiveSalinity;
constexpr DVL::Command DVL::cLiveTemperature;
constexpr DVL::Command DVL::cOrientationResolution;
constexpr DVL::Command DVL::cStaticMagneticHeadingOffset;
constexpr DVL::Command DVL::cCoordinateTransformation;
constexpr DVL::Command DVL::cDopplerParameterSource;
constexpr DVL::Command DVL::cSensorSource;

constexpr DVL::Command DVL::cRecorderErase;
constexpr DVL::Command DVL::cShowMemoryUsage;
constexpr DVL::Command DVL::cSetFileName;
constexpr DVL::Command DVL::cRecorderDisable;
constexpr DVL::Command DVL::cRecorderEnable;
constexpr DVL::Command DVL::cFileDownload;

constexpr DVL::Command DVL::cPreDeploymentTests;
constexpr DVL::Command DVL::cInteractiveTest;
constexpr DVL::Command DVL::cSendRawWaterCurrentData;
constexpr DVL::Command DVL::cSendWithoutSensorData;
constexpr DVL::Command DVL::cSendWithSensorData;
constexpr DVL::Command DVL::cSendASCIIData;
constexpr DVL::Command DVL::cDisplaySystemConfiguration;
constexpr DVL::Command DVL::cDisplayFixedLeader;
constexpr DVL::Command DVL::cDisplayTransformMatrix;
constexpr DVL::Command DVL::cDisplayPingSequence;
constexpr DVL::Command DVL::cTestDVLRecievePath;
constexpr DVL::Command DVL::cTestTxRxLoop;
constexpr DVL::Command DVL::cTestWiring;
constexpr DVL::Command DVL::cTestDVLRecievePathRepeat;
constexpr DVL::Command DVL::cTestTxRxLoopRepeat;
constexpr DVL::Command DVL::cTestWiringRepeat;
constexpr DVL::Command DVL::cTestAll;
constexpr DVL::Command DVL::cTestAllRepeat;

constexpr DVL::Command DVL::cSendSensorCommand;
constexpr DVL::Command DVL::cSensorDataOut;
constexpr DVL::Command DVL::cAuxSensorAuxMenu;
constexpr DVL::Command DVL::cPressureSensorOffset;
constexpr DVL::Command DVL::cSensorPortAssignment;
constexpr DVL::Command DVL::cSensorReset;

constexpr DVL::Command DVL::cTimePerEnsemble;
constexpr DVL::Command DVL::cTimeBetweenPings;
constexpr DVL::Command DVL::cSetRealTimeClock;

constexpr DVL::Command DVL::cFalseTargetThreshold;
constexpr DVL::Command DVL::cBandwidthMode;
constexpr DVL::Command DVL::cCorrelationThreshold;
constexpr DVL::Command DVL::cDataOut;
constexpr DVL::Command DVL::cErrorVelocityThreshold;
constexpr DVL::Command DVL::cBlankingDistance;
constexpr DVL::Command DVL::cReceiverGain;
constexpr DVL::Command DVL::cNumberOfBins;
constexpr DVL::Command DVL::cPingsPerEnsemble;
constexpr DVL::Command DVL::cBinSize;
constexpr DVL::Command DVL::cTransmitLength;
constexpr DVL::Command DVL::cAmbiguityVelocity;

constexpr DVL::Command DVL::cListFeatures;
constexpr DVL::Command DVL::cInstallFeature;

constexpr DVL::frameid_t DVL::kPD0HeaderID;
constexpr DVL::frameid_t DVL::kPD0FixedLeaderID;
constexpr DVL::frameid_t DVL::kPD0VariableLeaderID;
constexpr DVL::frameid_t DVL::kPD0VelocityDataID;
constexpr DVL::frameid_t DVL::kPD0CorrelationMagnitudeID;
constexpr DVL::frameid_t DVL::kPD0EchoIntensityID;
constexpr DVL::frameid_t DVL::kPD0PercentGoodID;
constexpr DVL::frameid_t DVL::kPD0StatusDataID;
constexpr DVL::frameid_t DVL::kPD0BottomTrackID;
constexpr DVL::frameid_t DVL::kPD0EnvironmentID;
constexpr DVL::frameid_t DVL::kPD0BottomTrackCommandID;
constexpr DVL::frameid_t DVL::kPD0BottomTrackHighResID;
constexpr DVL::frameid_t DVL::kPD0BottomTrackRangeID;
constexpr DVL::frameid_t DVL::kPD0SensorDataID;
constexpr DVL::frameid_t DVL::kPD4HeaderID;
constexpr DVL::frameid_t DVL::kPD5HeaderID;

constexpr DVL::DVLSpeed DVL::k300;
constexpr DVL::DVLSpeed DVL::k1200;
constexpr DVL::DVLSpeed DVL::k2400;
constexpr DVL::DVLSpeed DVL::k4800;
constexpr DVL::DVLSpeed DVL::k9600;
constexpr DVL::DVLSpeed DVL::k19200;
constexpr DVL::DVLSpeed DVL::k38400;
constexpr DVL::DVLSpeed DVL::k57600;
constexpr DVL::DVLSpeed DVL::k115200;











