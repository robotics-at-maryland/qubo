

void DVL::loadFactorySettings() {
    writeCommand(cRetrieveFactoryParameters);
    readMessage();
}

void DVL::loadUserSettings() {
    writeCommand(cRetrieveUserParameters);
    readMessage();
}

void DVL::saveUserSettings() {
    writeCommand(cKeepParameters);
    readMessage();
}

std::string DVL::getSystemInfo() {
    return new std::string(sendMessage(cDistplaySystemConfiguration).text);
}

std::string DVL::getFeaturesList() {
    return new std::string(sendMessage(cListFeatures).text);
}

std::string DVL::getBottomTrackHelp() {
    return new std::string(sendCommand(cBottomControlHelp).text);
}

std::string DVL::getControlHelp() {
    return new std::string(sendCommand(cControlCommandHelp).text);
}

std::string DVL::getEnvironmentHelp() {
    return new std::string(sendCommand(cEnvironmentCommandHelp).text);
}

std::string DVL::getLoopRecorderHelp() {
    return new std::string(sendCommand(cLoopRecorderCommandHelp).text);
}

std::string DVL::getPerformanceTestHelp() {
    return new std::string(sendCommand(cPerformanceTestCommandHelp).text);
}

std::string DVL::getSensorCommandHelp() {
    return new std::string(sendCommand(cSensorCommandHelp).text);
}

std::string DVL::getTimeCommandHelp() {
    return new std::string(sendCommand(cTimeCommandHelp).text);
}

std::string DVL::getWaterProfilingHelp() {
    return new std::string(sendCommand(cWaterProfilingCommandHelp).text);
}

void setSystemConfiguration(SystemConfig& config) {
    sendCommand(cSerialPortControl, 
            config.speed.id, 
            config.parity, 
            config.two_stopbits ? 1 : 0);
    sendCommand(cFlowControl, 
            config.auto_ensemble_cycling ? 1 : 0,
            config.auto_ping_cycling ? 1 : 0,
            config.binary_data_output ? 1 : 0,
            config.serial_output ? 1 : 0);
    sendCommand(config.turnkey ? cTurnkeyEnable : cTurnkeyDisable);
    sendCommand(config.recorder_enable ? cRecorderEnable : cRecorderDisable);
}

void DVL::setVehicleConfiguration(VehicleConfig& config) {
    sendCommand(cStaticHeadingOffset, config.beam3_alignment);
    sendCommand(cCoordinateTransformation, 
            config.transformation,
            config.use_pitch_roll ? 1 : 0,
            config.use_three_beams ? 1 : 0,
            config.use_bin_mappings ? 1 : 0);
    sendCommand(cOrientationResolution, config.orientation);
    sendCommand(cSensorSource, 
            config.condition_sources[0],
            config.condition_sources[1],
            config.condition_sources[2],
            config.condition_sources[3],
            config.condition_sources[4],
            config.condition_sources[5],
            config.condition_sources[6],
            config.condition_sources[7]);
    sendCommand(cSensorPortAssignment,
            2,
            config.port2);
    sendCommand(cSensorPortAssignment,
            3,
            config.port3);
    sendCommand(cSensorPortAssignment,
            4,
            config.port4);
}

void DVL::setLiveConditions(LiveConditions& conditions) {
    sendCommand(cLiveDepth, conditions.depth);
    sendCommand(cLiveHeading, conditions.heading);
    sendCommand(cLivePitchAndRoll, 
            conditions.pitch, 
            conditions.roll, 
            1); // 1 for ship coords, 0 for inst coords
    sendCommand(cLiveSalinity, conditions.salinity);
    sendCommand(cLiveTemperature, conditions.temperature);
}

void DVL::setDataTransferConfiguration(DataConfig& config) {
    switch (config) {
        case ALL_DATA:
            sendCommand(cSendRawWaterCurrentData);
            for (i = HPR_GYRO; i < SENSOR_LAST; i++) {
                sendCommand(cSensorDataOut,
                    i,
                    sensor_output[i][0],
                    sensor_output[i][1],
                    sensor_output[i][2],
                    sensor_output[i][3]);
            }
            sendCommand(cBottomDataOutput,
                    bottom_output[TRACK],
                    bottom_output[COMMAND],
                    bottom_output[HIGHRES],
                    bottom_output[RANGE],
                    bottom_output[NAVIGATION]);
            sendCommand(cDataOut,
                    profile_output[VELOCITY],
                    profile_output[CORRELATION],
                    profile_output[ECHO_INTENSITY],
                    profile_output[PERCENT_GOOD],
                    profile_output[STATUS]);
            break;
        case PARTIAL_DATA:
            sendCommand(cSendWithSensorData);
            break;
        case MINIMUM_DATA:
            sendCommand(cSendWithoutSensorData);
            break;
        case TEXT_DATA:
            sendCommand(cSendASCIIData);
            break;
    }
}

void setBottomTrackConfiguration(BottomTrackConfig& config) {
    sendCommand(cBottomTrackPingsPerEnsemble, config.pings_per_ensemble);
    sendCommand(cMaximumTrackingDepth, config.maximum_depth);
    sendCommand(cEvaluationAmplitudeMinimum, config.evaluation_amplitude_minimum);
    sendCommand(cBottomBlankingInterval, config.bottom_blank_interval);
    sendCommand(cBottomCorrelationMagnitude, config.correlation_magnitude_minimum);
    sendCommand(cBottomErrorVelocityMaximum, config.error_velocity_maximum);
    sendCommand(cBottomDepthGuess, config.depth_guess);
    sendCommand(cBottomGainSwitchDepth, config.gain_switch_depth);
    sendCommand(cWaterMassPing, config.water_mass_mode);
    sendCommand(cWaterMassParameters, 
            config.water_mass_layer_size,
            config.water_mass_near_bound,
            config.water_mass_far_bound);
    sendCommand(cSpeedLogControl,
            config.distance_kept_while_unlocked,
            config.distance_timeout_while_unlocked);
    sendCommand(cDistanceMeasureFilterConstant, config.distance_filter_constant);
}

void setWaterProfileConfiguration(WaterProfileConfig& config) {
    sendCommand(cBandwithMode, config.narrow_bandwidth);
    sendCommand(cBlankingDistance, config.blank_after_transmit);
    sendCommand(cNumberOfBins, config.depth_cells);
    sendCommand(cPingsPerEnsemble, config.pings_per_ensemble);
    sendCommand(cBinSize, config.depth_cell_size);
    sendCommand(cAmbiguityVelocity, config.radial_ambiguity_velocity);
    sendCommand(cFalseTargetThreshold, config.false_target_threshold);
    sendCommand(cCorrelationThreshold, config.low_correlation_threshold);
    sendCommand(cErrorVelocityThreshold, config.error_velocity_threshold);
    sendCommand(cRecieverGain, config.high_gain);
    sendCommand(cTransmitLength, config.transmit_length);
}

void setRecorderFilename(std::string& name) {
    sendCommand(cSetFileName, name.c_str());
}

void eraseRecorder() {
    sendCommand(cRecorderErase);
}

std::string getRecorderStatus() {
    return new std::string(sendCommand(cShowMemoryUsage).text);
}

void getRecordedData() {
    throw new std::exception("NOT IMPLEMENTED");
}

std::string getInstrumentTransformationMatrix() {
    return new std::string(sendCommand(cDisplayTransformMatrix).text);
}

std::string getPingSequence() {
    return new std::string(sendCommand(cDisplayPingSequence).text);
}

std::string testPreDeployment() {
    return new std::string(sendCommand(cPreDeploymentTests).text);
}

std::string testReceiver() {
    return new std::string(sendCommand(cTestDVLRecievePath).text);
}

std::string testContinuity() {
    return new std::string(sendCommand(cTestTxRxLoop).text);
}

std::string testWiring() {
    return new std::string(sendCommand(cTestWiring).text);
}

void enableMeasurement() {
    sendCommand(cStartPinging);
}

void disableMeasurement() {
    sendBreak();
}

DVLData getDVLData() {
    Message message = readMessage();
    DVLData data = {0};
    data.mms_east = message.pd0_bottom_track->bot_velocity[0];
    data.mms_north = message.pd0_bottom_track->bot_velocity[1];
    data.mms_surface = message.pd0_bottom_track->bot_velocity[2];
}


































