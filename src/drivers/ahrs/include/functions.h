
public: // Front facing API function calls.

/** 
 * Reads the hardware info from the AHRS unit. 
 * @return (std::string) Hardware info
 */
std::string getInfo();

/**
 * Set the declination angle used to determine true north.
 * This is not used if only magnetic north is needed.
 * This is an outside variable depending on global location.
 * @param (float32) -180 to +180 degrees.
 */
void setDeclination(float decl);

/**
 * Get the declination angle used to determine true north.
 * This is not used if only magnetic north is needed.
 * This is an outside variable depending on global location.
 * @return (float32) -180 to +180 degrees.
 */
float getDeclination();

/**
 * Set the true north flag.
 * If true north is enabled, accuracy depends on the declination.
 * Otherwise the magnetic north is used without correction.
 * @param (bool) true for true north, false for magnetic.
 */
void setTrueNorth(bool north);

/**
 * Get the true north flag.
 * If true north is enabled, accuracy depends on the declination.
 * Otherwise the magnetic north is used without correction.
 * @return (bool) true for true north, false for magnetic.
 */
bool getTrueNorth();

/**
 * Set the endian-ness of the data I/O.
 * Little-endian is used in linux/unix machines.
 * Big-endian is used in TRAX Studio/elsewhere.
 * @param (bool) true for big-endian, false for little-endian.
 */
void setBigEndian(bool endian);

/**
 * Get the endian-ness of the the data I/O.
 * Little-endian is used in linux/unix machines.
 * Big-endian is used in TRAX Studio/elsewhere.
 * @return (bool) true for big-endian, false for little-endian.
 */
bool getBigEndian();

/**
 * Set the mounting reference orientation.
 * @param (MountRef)
 */
void setMounting(MountRef mount);

/**
 * Get the mounting reference orientation.
 * @return (MountRef)
 */
MountRef getMounting();

/**
 * Set the number of points in a user calibration.
 * @param (int) Number of points to expect during calibration.
 */
void setCalPoints(unsigned int points);

/**
 * Get the number of points in a user calibration.
 * @return (unsigned int) Number of points to expect during calibration.
 */
unsigned int getCalPoints();

/**
 * Set the auto-calibration flag.
 * Auto-calibration automatically acquires a point if it is suitable.
 * Manual calibration allows the user to choose each exact point.
 * @param (bool) true for auto, false for manual.
 */
void setAutoCalibration(bool cal);

/**
 * Get the auto-calibration flag.
 * Auto-calibration automatically acquires a point if it is suitable.
 * Manual calibration allows the user to choose each exact point.
 * @return (bool) true for auto, false for manual.
 */
bool getAutoCalibration();

/** 
 * Set the baudrate to the device.
 * NOTE: change requires a full power-cycle to update.
 * @param (AHRSSpeed) Speed to use on the connection.
 */
void setBaudrate(AHRSSpeed speed);

/** 
 * Get the baudrate to the device.
 * @return (AHRSSpeed) Speed to use on the connection.
 */
AHRSSpeed getBaudrate();

/**
 * Set the Mils/Degree flag.
 * 6400 Mils/circle
 * 360 Degrees/circle
 * @param (bool) true for mils, false for degrees.
 */
void setMils(bool mils);

/**
 * Get the Mils/Degree flag.
 * 6400 Mils/circle
 * 360 Degrees/circle
 * @return (bool) true for mils, false for degrees.
 */
bool getMils();

/**
 * Set Heading/Pitch/Roll calibration flag.
 * If true, heading pitch and roll data is sent 
 * while the device is being calibrated.
 * @param (bool) true for output, false for no data.
 */
void setHPRCal(bool hpr);

/**
 * Get Heading/Pitch/Roll calibration flag.
 * If true, heading pitch and roll data is sent 
 * while the device is being calibrated.
 * @return (bool) true for output, false for no data.
 */
bool getHPRCal();

/**
 * Set the current magnetometer calibration id.
 * The device stores up to 8 different calibrations.
 * @param (CalibrationID)
 */
void setMagCalID(CalibrationID id);

/**
 * Get the current magnetometer calibration id.
 * The device stores up to 8 different calibrations.
 * @return (CalibrationID)
 */
CalibrationID getMagCalID();

/**
 * Set the current acclerometer calibration id.
 * The device stores up to 8 different calibrations.
 * @param (CalibrationID)
 */
void setAccelCalID(CalibrationID id);

/**
 * Get the current acclerometer calibration id.
 * The device stores up to 8 different calibrations.
 * @return (CalibrationID)
 */
CalibrationID getAccelCalID();

/**
 * Set the magnetic truth method.
 * @param (TruthMethod)
 */
void setMagTruthMethod(TruthMethod method);

/**
 * Get the magnetic truth method.
 * @param (TruthMethod)
 */
TruthMethod getMagTruthMethod();

/**
 * Saves the current configuration to non-volatile memory.
 * This persists across hard restarts and power loss.
 */
void saveConfig();

/**
 * Reset the magnetic field reference.
 * Aligns the HPR to the magnetometer/accel heading.
 * Use only when the device is stable and the magnetic field
 * is not changing significantly.
 */
void resetMagReference();

/**
 * Set Aquisition Configuration
 * @param (AcqParams)
 */
void setAcqConfig(AcqConfig config);

/**
 * Get Aquisition Parameters.
 * @return (AcqParams)
 */
AcqConfig getAcqConfig();

/** Send the data poll format to the AHRS. */
void sendAHRSDataFormat();

/** 
 * Polls the AHRS for position information.
 * @return an AHRSData struct of formatted AHRS data.
 */
AHRSData pollAHRSData();

/**
 * Start User Calibration.
 * @param (CalType) type of calibration.
 */
void startCalibration(CalType);

/** Stop User Calibration. */
void stopCalibration();

/**
 * Take User Calibration Point.
 * @return (int) current cal point number.
 */
int takeCalibrationPoint();

/**
 * Get the calibration score for the calibration just completed.
 * @return (CalScore)
 */
CalScore getCalibrationScore();

/** Reset the magnetometer to factory calibration settings. */
void resetMagCalibration();

/** Reset the accelerometer to factory calibration settings. */
void resetAccelCalibration();

/**
 * Set the AHRS mode flag.
 * @param (bool) true for AHRS, false for Compass.
 */
void setAHRSMode(bool ahrs);

/**
 * Get the AHRS mode flag.
 * @param (bool) true for AHRS, false for Compass.
 */
bool getAHRSMode();

/**
 * Set the Finite-Impuse-Response filters coefficient array.
 * @param (FilterData) filter configuration
 */
void setFIRFilters(FilterData filters);

/**
 * Get the Finite-Impuse-Response filters coefficient array.
 * @return (FilterData) filter configuration
 */
FilterData getFIRFilters();

/** Power down the device to conserve power. */
void powerDown();

/** Wake up the device from sleep. */
void wakeUp();

