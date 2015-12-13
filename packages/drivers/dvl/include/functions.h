
public: // Front facing API function calls.

void loadFactorySettings();

void loadUserSettings();

void saveUserSettings();

std::string getSystemInfo();

std::string getFeaturesList();

std::string getBottomTrackHelp();

std::string getControlHelp();

std::string getEnvironmentHelp();

std::string getLoopRecorderHelp();

std::string getPerformanceTestHelp();

std::string getSensorCommandHelp();

std::string getTimeCommandHelp();

std::string getWaterProfilingHelp();

void setSystemConfiguration(SystemConfig& config);

void setVehicleConfiguration(VehicleConfig& config);

void setLiveConditions(LiveConditions& conditions);

void setDataTransferConfiguration(DataConfig& config);

void setBottomTrackConfiguration(BottomTrackConfig& config);

void setWaterProfileConfiguration(WaterProfileConfig& config);

void setRecorderFilename(std::string name);

void setRecorderEnable(bool enable);

void eraseRecorder();

std::string getRecorderStatus();

FILE getRecordedData();

std::string getInstrumentTransformationMatrix();

std::string getPingSequence();

std::string testPreDeployment();

std::string testReceiver();

std::string testContinuity();

std::string testWiring();

DVLData getDVLData();











