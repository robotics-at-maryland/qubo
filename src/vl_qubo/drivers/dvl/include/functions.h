
public: // Front facing API function calls.

/** Load the known-good settings from the factory. */
void loadFactorySettings();

/** Load the saved user settings */
void loadUserSettings();

/** 
 * Save the current user settings to non-volatile memory
 * allowing it to be recalled with loadUserSettings()
 */
void saveUserSettings();

/**
 * Get the hardware system info from the DVL
 * @return (std::string) of formatted hardware info
 */
std::string getSystemInfo();

/**
 * Get the current hardware/software features avaliable on the DVL.
 * @return (std::string) of formatted features.
 */
std::string getFeaturesList();

/**
 * Get help with bottom tracking commands
 * Irrelevant to this API specification.
 * @return (std::string) of a formatted help menu
 */
std::string getBottomTrackHelp();

/**
 * Get help with control commands
 * Irrelevant to this API specification.
 * @return (std::string) of a formatted help menu
 */
std::string getControlHelp();

/**
 * Get help with environmental commands
 * Irrelevant to this API specification.
 * @return (std::string) of a formatted help menu
 */
std::string getEnvironmentHelp();

/**
 * Get help with loop recorder commands
 * Irrelevant to this API specification.
 * @return (std::string) of a formatted help menu
 */
std::string getLoopRecorderHelp();

/**
 * Get help with performance testing commands
 * Irrelevant to this API specification.
 * @return (std::string) of a formatted help menu
 */
std::string getPerformanceTestHelp();

/**
 * Get help with sensor commands
 * Irrelevant to this API specification.
 * @return (std::string) of a formatted help menu
 */
std::string getSensorCommandHelp();

/**
 * Get help with time commands
 * Irrelevant to this API specification.
 * @return (std::string) of a formatted help menu
 */
std::string getTimeCommandHelp();

/**
 * Get help with water profiling commands
 * Irrelevant to this API specification.
 * @return (std::string) of a formatted help menu
 */
std::string getWaterProfilingHelp();

/**
 * Set system configuration
 * This includes details about the communication protocol and
 * data transfers over the serial connection.
 * @param (SystemConfig) struct with system configuration details.
 */
void setSystemConfiguration(SystemConfig& config);

/**
 * Set the vehicle/device configuration
 * This includes details about the relative position of the device,
 * Sensor serial configuration, and data source configuration.
 * @param (VehicleConfig) struct with vehicle configuration details.
 */
void setVehicleConfiguration(VehicleConfig& config);

/**
 * Set the live conditions of the vehicle.
 * These are used in order to provide an earth reference and to more 
 * accurately calculate the velocity data. This includes data such as device 
 * heading, pitch and roll, as well as details pertaining to the current 
 * speed of sound, such as salinity, depth, and temperature.
 * @param (LiveConditions) struct with live condition details.
 */
void setLiveConditions(LiveConditions& conditions);

/**
 * Set the data transfer configuration.
 * Thse are details about what data needs to be transferred back after each 
 * collection cycle is complete. This includes the choice of full/partial 
 * output data, and what data to include in the full data report.
 * @param (DataConfig) struct with data output configuration details.
 */
void setDataTransferConfiguration(DataConfig& config);

/**
 * Set the bottom tracking configuration.
 * This is data pertaining to detecting and measuring velocity relative to 
 * the solid bottom of the water, and referencing a mass of water some 
 * distance off of the bottom.
 * @param (BottomTrackConfig) struct with bottom tracking configuration.
 */
void setBottomTrackConfiguration(BottomTrackConfig& config);

/**
 * Set the water profiling configuration.
 * This is data about measuring the motion of the water in the water-mass 
 * ping measurements, and settings regarding collection and integrity of the 
 * data collected.
 * @param (WaterProfileConfig) struct of water profiling configuration details.
 */
void setWaterProfileConfiguration(WaterProfileConfig& config);

/**
 * Set the file name of the on-device logfile.
 * This file can be downloaded later.
 * @param (std::string) of length <= 32 for a filename.
 */
void setRecorderFilename(std::string& name);

/** Erase the recorder's data file completely. */
void eraseRecorder();

/**
 * Get the log file status.
 * This is a file size/usage message in a formatted string.
 * @return (std::string) a formatted string of usage data.
 */
std::string getRecorderStatus();

/**
 * Get the log file from the device.
 * This is transmitted over the YMODEM protocol. A 2mb file (max size) could 
 * take nearly 20s on the fastest (115kbaud) connection.
 * NOTE: INCOMPLETE
 */
void getRecordedData();

/**
 * Get the transform matrix.
 * This matrix tranforms from device coordinates to vehicle coords.
 * @return (std::string) a formatted matrix of values.
 */
std::string getInstrumentTransformationMatrix();

/**
 * Get the water-mass/bottom-track ping sequence
 * String contains a sequence of W and B characters displaying the order 
 * of pings sent to measure the (W)ater and (B)ottom conditions.
 * @return (std::string) a formatted order of pings.
 */
std::string getPingSequence();

/**
 * Run pre-deployment tests
 * This is a system self-check to confirm that the device is within
 * operation parameters. This tests RAM, ROM, and some details tested under 
 * other tests. This also produces a go/no-go composite result at the end.
 * This test is meant to be run with the sensor in the water.
 * @return (std::string) a formatted list of test results.
 */
std::string testPreDeployment();

/**
 * Test the receiver electronics.
 * These are electrical system checks for the sensor.
 * This includes correlation magnitude, duty cycle, and signal to noise.
 * @return (std::string) a formatted string of test results.
 */
std::string testReceiver();

/**
 * Test the continuity of the whole transmitter-transducer-receiver system.
 * Checks acoustic continuity from transmitter to receiver. 
 * This includes noise, electronic, and ceramic tests.
 * This test is meant to be run with the sensor in the water.
 * @return (std::string) a formatted string of test results.
 */
std::string testContinuity();

/**
 * Test the wiring between the electronics enclosure and the sensor head.
 * This provides a somewhat details message to aid in troubleshooting.
 * This test should be run at 24V in order for the test result to be accurate.
 * @return (std::string) a formatted string of test results.
 */
std::string testWiring();

/**
 * Enable data collection.
 * After this command is called, only the disableMeasurement or getDVLData
 * commands should be executed. This is for reasons of data-integrity and
 * protocol restrictions regarding collection of data.
 */
void enableMeasurement();

/**
 * Disable data collection.
 * This command should be run after enableMeasurement when the user
 * wishes to stop collecting data from the DVL.
 */
void disableMeasurement();

/**
 * Poll the DVL for a data packet.
 * This must only be run after an enableMeasurement call, or it will
 * time-out and throw an exception when there is no data.
 * The data contained in this packet can change based on the type of data
 * polled from the DVL itself, based on the settings in the 
 * setDataTransferConfiguration command. One is advised to pay attention to
 * only the fields that contain known-good data, and to disregard the
 * undefined data that does not come from the DVL in each packet.
 */
DVLData getDVLData();


































