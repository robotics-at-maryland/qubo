

std::string AHRS::getInfo() {
   ModInfo info;
   sendCommand(kGetModInfo, NULL, kGetModInfoResp, &info);
   return std::string(((char*) &info), sizeof(ModInfo));
}

void AHRS::setDeclination(float decl) {
   ConfigFloat32 data = {kDeclination, decl};
   sendCommand(kSetConfigFloat32, &data, kSetConfigDone, NULL);
   saveConfig();
}

float AHRS::getDeclination() {
   ConfigFloat32 data;
   sendCommand(kGetConfig, &kDeclination, kGetConfigRespFloat32, &data);
   return data.value;
}

void AHRS::setTrueNorth(bool north) {
   ConfigBoolean data = {kTrueNorth, north};
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool AHRS::getTrueNorth() {
   ConfigBoolean data;
   sendCommand(kGetConfig, &kTrueNorth, kGetConfigRespBoolean, &data);
   return data.value;
}

void AHRS::setBigEndian(bool endian) {
   ConfigBoolean data = {kBigEndian, endian};
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool AHRS::getBigEndian() {
   ConfigBoolean data;
   sendCommand(kGetConfig, &kBigEndian, kGetConfigRespBoolean, &data);
   return data.value;
}

void AHRS::setMounting(MountRef mount) {
   ConfigUInt8 data = {kMountingRef, mount};
   sendCommand(kSetConfigUInt8, &data, kSetConfigDone, NULL);
   saveConfig();
}

AHRS::MountRef AHRS::getMounting() {
   ConfigUInt8 data;
   sendCommand(kGetConfig, &kMountingRef, kGetConfigRespUInt8, &data);
   switch(data.id) {
      case STD_0:       return STD_0;
      case X_UP_0:      return X_UP_0;
      case Y_UP_0:      return Y_UP_0;
      case STD_90:      return STD_90;
      case STD_180:     return STD_180;
      case STD_270:     return STD_270;
      case Z_DOWN_0:    return Z_DOWN_0;
      case X_UP_90:     return X_UP_90;
      case X_UP_180:    return X_UP_180;
      case X_UP_270:    return X_UP_270;
      case Y_UP_90:     return Y_UP_90;
      case Y_UP_180:    return Y_UP_180;
      case Y_UP_270:    return Y_UP_270;
      case Z_DOWN_90:   return Z_DOWN_90;
      case Z_DOWN_180:  return Z_DOWN_180;
      case Z_DOWN_270:  return Z_DOWN_270;
      default:          break;
   }
   throw AHRSException("Unknown mounting reference id reported.");
}

void AHRS::setCalPoints(unsigned int points) {
   ConfigUInt32 data = {kUserCalNumPoints, points};
   sendCommand(kSetConfigUInt32, &data, kSetConfigDone, NULL);
   saveConfig();
}

unsigned int AHRS::getCalPoints() {
   ConfigUInt32 data;
   sendCommand(kGetConfig, &kUserCalNumPoints, kGetConfigRespUInt32, &data);
   return data.value;
}

void AHRS::setAutoCalibration(bool cal) {
   ConfigBoolean data = {kUserCalAutoSampling, cal};
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool AHRS::getAutoCalibration() {
   ConfigBoolean data;
   sendCommand(kGetConfig, &kUserCalAutoSampling, kGetConfigRespBoolean, &data);
   return data.value;
}

void AHRS::setBaudrate(AHRSSpeed speed)
{
   ConfigUInt8 data = {kBaudRate, speed.id};
   sendCommand(kSetConfigUInt8, &data, kSetConfigDone, NULL);
   saveConfig();
}

AHRS::AHRSSpeed AHRS::getBaudrate() {
   ConfigUInt8 data;
   sendCommand(kGetConfig, &kBaudRate, kGetConfigRespUInt8, &data);
   switch (data.value) {
      case k0.id:       return k0;
      case k2400.id:    return k2400;
      case k4800.id:    return k4800;
      case k9600.id:    return k9600;
      case k19200.id:   return k19200;
      case k38400.id:   return k38400;
      case k57600.id:   return k57600;
      case k115200.id:  return k115200;
      default:          break;
   }
   throw AHRSException("Unknown device baudrate id reported.");
}

void AHRS::setMils(bool mils) {
   ConfigBoolean data = {kMilOut, mils};
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool AHRS::getMils() {
   ConfigBoolean data;
   sendCommand(kGetConfig, &kMilOut, kGetConfigRespBoolean, &data);
   return data.value;
}

void AHRS::setHPRCal(bool hpr) {
   ConfigBoolean data = {kHPRDuringCal, hpr};
   sendCommand(kSetConfigBoolean, &data, kSetConfigDone, NULL);
   saveConfig();
}

bool AHRS::getHPRCal() {
   ConfigBoolean data;
   sendCommand(kGetConfig, &kHPRDuringCal, kGetConfigRespBoolean, &data);
   return data.value;
}

void AHRS::setMagCalID(CalibrationID id) {
   ConfigUInt32 data = {kMagCoeffSet, id};
   sendCommand(kSetConfigUInt32, &data, kSetConfigDone, NULL);
   saveConfig();
}

AHRS::CalibrationID AHRS::getMagCalID() {
   ConfigUInt32 data;
   sendCommand(kGetConfig, &kMagCoeffSet, kGetConfigRespUInt32, &data);
   switch(data.value) {
      case CAL_0: return CAL_0;
      case CAL_1: return CAL_1;
      case CAL_2: return CAL_2;
      case CAL_3: return CAL_3;
      case CAL_4: return CAL_4;
      case CAL_5: return CAL_5;
      case CAL_6: return CAL_6;
      case CAL_7: return CAL_7;
      default:    break;
   }
   throw AHRSException("Unexpected device calibration id reported.");
}

void AHRS::setAccelCalID(CalibrationID id) {
   ConfigUInt32 data = {kAccelCoeffSet, id};
   sendCommand(kSetConfigUInt32, &data, kSetConfigDone, NULL);
   saveConfig();
}

AHRS::CalibrationID AHRS::getAccelCalID() {
   ConfigUInt32 data;
   sendCommand(kGetConfig, &kAccelCoeffSet, kGetConfigRespUInt32, &data);
   switch(data.value) {
      case CAL_0: return CAL_0;
      case CAL_1: return CAL_1;
      case CAL_2: return CAL_2;
      case CAL_3: return CAL_3;
      case CAL_4: return CAL_4;
      case CAL_5: return CAL_5;
      case CAL_6: return CAL_6;
      case CAL_7: return CAL_7;
      default:    break;
   }
   throw AHRSException("Unexpected device calibration id reported.");
}

void AHRS::setMagTruthMethod(TruthMethod method) {
   writeCommand(kSetMagTruthMethod, &method);
   saveConfig();
}

AHRS::TruthMethod AHRS::getMagTruthMethod() {
   MagTruthMethod data;
   sendCommand(kGetMagTruthMethod, NULL, kGetMagTruthMethodResp, &data);
   switch(data) {
      case STANDARD: return STANDARD;
      case TIGHT: return TIGHT;
      case AUTOMERGE: return AUTOMERGE;
      default:    break;
   }
   throw AHRSException("Unexpected device truth method reported.");
}

void AHRS::saveConfig() {
   SaveError err;
   sendCommand(kSave, NULL, kSaveDone, &err);
   if (err) 
      throw AHRSException("Error while saving configuration to nonvolatile memory.");
}

void AHRS::resetMagReference() {
   writeCommand(kSetResetRef, NULL);
}

void AHRS::setAcqConfig(AcqConfig config) {
   AcqParams params = {config.poll_mode, config.flush_filter, 0, config.sample_delay};
   sendCommand(kSetAcqParams, &params, kSetAcqParamsDone, NULL);
   saveConfig();
}

AHRS::AcqConfig AHRS::getAcqConfig() {
   AcqParams params;
   sendCommand(kGetAcqParams, NULL, kGetAcqParamsResp, &params);
   return { params.sample_delay,
            ((params.flush_filter) ? true : false), 
            ((params.aquisition_mode) ? true : false)};
}

void AHRS::sendAHRSDataFormat()
{
   setBigEndian(false);
   writeCommand(kSetDataComponents, &dataConfig);
   saveConfig();
}

AHRS::AHRSData AHRS::pollAHRSData()
{
   RawData data;
   // Poll the AHRS for a data message.
   sendCommand(kGetData, NULL, kGetDataResp, &data);
   // Copy all the data to the actual AHRS storage.
   _lastReading.quaternion[0] = data.quaternion[0];
   _lastReading.quaternion[1] = data.quaternion[1];
   _lastReading.quaternion[2] = data.quaternion[2];
   _lastReading.quaternion[3] = data.quaternion[3];
   _lastReading.gyroX = data.gyroX;
   _lastReading.gyroY = data.gyroY;
   _lastReading.gyroZ = data.gyroZ;
   _lastReading.accelX = data.accelX;
   _lastReading.accelY = data.accelY;
   _lastReading.accelZ = data.accelZ;
   _lastReading.magX = data.magX;
   _lastReading.magY = data.magY;
   _lastReading.magZ = data.magZ;
   return _lastReading;
}

void AHRS::startCalibration(CalType type) {
   writeCommand(kStartCal, &type);
}

void AHRS::stopCalibration() {
   writeCommand(kStopCal, NULL);
}

int AHRS::takeCalibrationPoint() {
   SampleCount point;
   sendCommand(kTakeUserCalSample, NULL, kUserCalSampleCount, &point);
   return point;
}

AHRS::CalScore AHRS::getCalibrationScore() {
   UserCalScore score;
   readCommand(kUserCalScore, &score);
   saveConfig();
   return { score.mag_cal_score, 
            score.accel_cal_score, 
            score.distribution_error, 
            score.tilt_error, 
            score.tilt_range};
}

void AHRS::resetMagCalibration() {
   sendCommand(kFactoryMagCoeff, NULL, kFactoryMagCoeffDone, NULL);
   saveConfig();
}

void AHRS::resetAccelCalibration() {
   sendCommand(kFactoryAccelCoeff, NULL, kFactoryAccelCoeffDone, NULL);
   saveConfig();
}

void AHRS::setAHRSMode(bool mode) {
   writeCommand(kSetFunctionalMode, &mode);
   saveConfig();
}

bool AHRS::getAHRSMode() {
   bool mode;
   sendCommand(kGetFunctionalMode, NULL, kGetFunctionalModeResp, &mode);
   return mode;
}

void AHRS::setFIRFilters(FilterData data) {
   char buffer[kSetFIRFiltersThirtyTwo.payload_size] = {3,1,(char) data.size()};
   char* begin = (char*) data.data();
   Command cmd;
   switch (data.size()) {
      case F_0:
         cmd = kSetFIRFiltersZero;
         break;
      case F_4:
         cmd = kSetFIRFiltersFour;
         break;
      case F_8:
         cmd = kSetFIRFiltersEight;
         break;
      case F_16:
         cmd = kSetFIRFiltersSixteen;
         break;
      case F_32:
         cmd = kSetFIRFiltersThirtyTwo;
         break;
      default: throw AHRSException("Invalid number of FIR filter coefficients!");
   }
   if (!memcpy(buffer+3, begin, data.size()*sizeof(double)))
      throw AHRSException("Could not copy filter data");
   sendCommand(cmd, buffer, kSetFIRFiltersDone, NULL);
}

AHRS::FilterData AHRS::getFIRFilters() {
   FIRFilter filter_id = {3,1};
   Message message;
   FilterData data;

   writeCommand(kGetFIRFilters, &filter_id);
   do {
      message = readMessage();
   } while (inferCommand(message).id != kGetFIRFiltersRespZero.id);

   data.resize((message.payload_size - 3)/sizeof(double));
   if (!memcpy(data.data(), message.payload->data() + 3, message.payload_size - 3))
      throw AHRSException("Could not copy filter data.");

   return data;
}

void AHRS::powerDown() {
   sendCommand(kPowerDown, NULL, kPowerDownDone, NULL);
}

void AHRS::wakeUp() {
   bool mode;
   writeCommand(kGetFunctionalMode, NULL);
   readCommand(kPowerUpDone, NULL);
   readCommand(kGetFunctionalMode, &mode);
}
