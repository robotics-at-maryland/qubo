

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
    return std::string(sendCommand(cDisplaySystemConfiguration).text);
}

std::string DVL::getFeaturesList() {
    return std::string(sendCommand(cListFeatures).text);
}

std::string DVL::getBottomTrackHelp() {
    return std::string(sendCommand(cBottomCommandHelp).text);
}

std::string DVL::getControlHelp() {
    return std::string(sendCommand(cControlCommandHelp).text);
}

std::string DVL::getEnvironmentHelp() {
    return std::string(sendCommand(cEnvironmentCommandHelp).text);
}

std::string DVL::getLoopRecorderHelp() {
    return std::string(sendCommand(cLoopRecorderCommandHelp).text);
}

std::string DVL::getPerformanceTestHelp() {
    return std::string(sendCommand(cPerformanceTestCommandHelp).text);
}

std::string DVL::getSensorCommandHelp() {
    return std::string(sendCommand(cSensorCommandHelp).text);
}

std::string DVL::getTimeCommandHelp() {
    return std::string(sendCommand(cTimeCommandHelp).text);
}

std::string DVL::getWaterProfilingHelp() {
    return std::string(sendCommand(cWaterProfilingCommandHelp).text);
}

void DVL::setSystemConfiguration(SystemConfig& config) {
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
    switch (config.output_type) {
        case ALL_DATA:
            //int i;
            sendCommand(cSendRawWaterCurrentData);
            /* this needs to be dynamic for each sensor.
            for (i = HPR_GYRO; i < SENSOR_LAST; i++) {
                sendCommand(cSensorDataOut,
                    i,
                    config.sensor_output[i][0],
                    config.sensor_output[i][1],
                    config.sensor_output[i][2],
                    config.sensor_output[i][3]);
            }
            */
            sendCommand(cBottomDataOut,
                    config.bottom_output[TRACK],
                    config.bottom_output[COMMAND],
                    config.bottom_output[HIGHRES],
                    config.bottom_output[RANGE],
                    config.bottom_output[NAVIGATION]);
            sendCommand(cDataOut,
                    config.profile_output[VELOCITY],
                    config.profile_output[CORRELATION],
                    config.profile_output[ECHO_INTENSITY],
                    config.profile_output[PERCENT_GOOD],
                    config.profile_output[STATUS]);
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

void DVL::setBottomTrackConfiguration(BottomTrackConfig& config) {
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

void DVL::setWaterProfileConfiguration(WaterProfileConfig& config) {
    sendCommand(cBandwidthMode, config.narrow_bandwidth);
    sendCommand(cBlankingDistance, config.blank_after_transmit);
    sendCommand(cNumberOfBins, config.depth_cells);
    sendCommand(cPingsPerEnsemble, config.pings_per_ensemble);
    sendCommand(cBinSize, config.depth_cell_size);
    sendCommand(cAmbiguityVelocity, config.radial_ambiguity_velocity);
    sendCommand(cFalseTargetThreshold, config.false_target_threshold);
    sendCommand(cCorrelationThreshold, config.low_correlation_threshold);
    sendCommand(cErrorVelocityThreshold, config.error_velocity_threshold);
    sendCommand(cReceiverGain, config.high_gain);
    sendCommand(cTransmitLength, config.transmit_length);
}

void DVL::setRecorderFilename(std::string& name) {
    sendCommand(cSetFileName, name.c_str());
}

void DVL::eraseRecorder() {
    sendCommand(cRecorderErase);
}

std::string DVL::getRecorderStatus() {
    return std::string(sendCommand(cShowMemoryUsage).text);
}

void DVL::getRecordedData() {
}

std::string DVL::getInstrumentTransformationMatrix() {
    return std::string(sendCommand(cDisplayTransformMatrix).text);
}

std::string DVL::getPingSequence() {
    return std::string(sendCommand(cDisplayPingSequence).text);
}

std::string DVL::testPreDeployment() {
    return std::string(sendCommand(cPreDeploymentTests).text);
}

std::string DVL::testReceiver() {
    return std::string(sendCommand(cTestDVLRecievePath).text);
}

std::string DVL::testContinuity() {
    return std::string(sendCommand(cTestTxRxLoop).text);
}

std::string DVL::testWiring() {
    return std::string(sendCommand(cTestWiring).text);
}

void DVL::enableMeasurement() {
    sendCommand(cTimePerEnsemble, 0,0,0);
    sendCommand(cTimeBetweenPings, 0,0,0);
    sendCommand(cStartPinging);
}

void DVL::disableMeasurement() {
    sendBreak();
}

DVL::DVLData DVL::getDVLData() {
    Message message = readMessage();
    DVLData data = {};
    int scanned;
    switch (message.format) {
        case FORMAT_PD0:
            data.transform = (CoordinateSystem) 
                ((message.pd0_fixed->coord_transform >> 3) & 0x3);
            if (message.pd0_velocity) {
                data.water_vel[0] = message.pd0_velocity[0][0];
                data.water_vel[1] = message.pd0_velocity[0][1];
                data.water_vel[2] = message.pd0_velocity[0][2];
                data.water_vel[3] = message.pd0_velocity[0][3];
            }
            if (message.pd0_bottom_track_highres) {
                data.bottom_vel[0] = message.pd0_bottom_track_highres->bot_velocity[0];
                data.bottom_vel[1] = message.pd0_bottom_track_highres->bot_velocity[1];
                data.bottom_vel[2] = message.pd0_bottom_track_highres->bot_velocity[2];
                data.bottom_vel[3] = message.pd0_bottom_track_highres->bot_velocity[3];
            }
            break;
        case FORMAT_PD4:
            data.transform = (CoordinateSystem) (message.pd4_data->system_config >> 6);
            data.water_vel[0] = message.pd4_data->velocity[0];
            data.water_vel[1] = message.pd4_data->velocity[1];
            data.water_vel[2] = message.pd4_data->velocity[2];
            data.water_vel[3] = message.pd4_data->velocity[3];
            data.bottom_vel[0] = message.pd4_data->bottom[0];
            data.bottom_vel[1] = message.pd4_data->bottom[1];
            data.bottom_vel[2] = message.pd4_data->bottom[2];
            data.bottom_vel[3] = message.pd4_data->bottom[3];
            break;
        case FORMAT_PD5:
            data.transform = (CoordinateSystem) (message.pd5_data->system_config >> 6);
            data.water_vel[0] = message.pd5_data->velocity[0];
            data.water_vel[1] = message.pd5_data->velocity[1];
            data.water_vel[2] = message.pd5_data->velocity[2];
            data.water_vel[3] = message.pd5_data->velocity[3];
            data.bottom_vel[0] = message.pd5_data->bottom[0];
            data.bottom_vel[1] = message.pd5_data->bottom[1];
            data.bottom_vel[2] = message.pd5_data->bottom[2];
            data.bottom_vel[3] = message.pd5_data->bottom[3];
            break;
        case FORMAT_PD6:
            /* These are the basic ways to grab data from the PD6 format.
            scanned = sscanf(message.pd6_attitude, 
                    ":SA,%f,%f,%f", 
                    &pitch, &roll, &heading);
            scanned = sscanf(message.pd6_timing, 
                    ":TS,%2d%2d%2d%2d%2d%2d%2d,%f,%f,%f,%f,%1d%x",
                    &year, &month, &day, &hour, &minute, &second, &hundreth,
                    &salinity, &temperature, &depth, &speedofsound, &nerrors, &errcode);
            if (message.pd6_w_instrument) {
                scanned = sscanf(message.pd6_w_instrument, 
                        ":WI,%d,%d,%d,%d,%c",
                        &xvel, &yvel, &zvel, &evel, &inststatus);
                scanned = sscanf(message.pd6_w_ship, 
                        ":WS,%d,%d,%d,%c",
                        &transverse, &longitudinal, &normal, &shipstatus);
                scanned = sscanf(message.pd6_w_earth, 
                        ":WE,%d,%d,%d,%c",
                        &east, &north, &up, &earthstatus);
                scanned = sscanf(message.pd6_w_distance,
                        ":WD,%f,%f,%f,%f,%f",
                        &east, &north, &up, &range, &time);
            }
            if (message.pd6_b_instrument) {
                    scanned = sscanf(message.pd6_b_instrument, 
                        ":BI,%d,%d,%d,%d,%c",
                        &xvel, &yvel, &zvel, &evel, &inststatus);
                    scanned = sscanf(message.pd6_b_ship, 
                        ":BS,%d,%d,%d,%c",
                        &transverse, &longitudinal, &normal, &shipstatus);
                    scanned = sscanf(message.pd6_b_earth, 
                        ":BE,%d,%d,%d,%c",
                        &east, &north, &up, &earthstatus);
                    scanned = sscanf(message.pd6_b_distance,
                        ":BD,%f,%f,%f,%f,%f",
                        &east, &north, &up, &range, &time);
            }
            */
            data.transform = CoordinateSystem::SHIP_COORD;
            scanned = sscanf(message.pd6_w_ship, 
                    ":WS,%d,%d,%d,%*c",
                    &(data.water_vel[0]), &(data.water_vel[1]), &(data.water_vel[2]));
            if (scanned != 3)
                throw new DVLException("Unable to parse PD6 Water Data");
            scanned = sscanf(message.pd6_b_ship, 
                    ":BS,%d,%d,%d,%*c",
                    &(data.bottom_vel[0]), &(data.bottom_vel[1]), &(data.bottom_vel[2]));
            if (scanned != 3)
                throw new DVLException("Unable to parse PD6 Bottom Data");
            break;
        case FORMAT_EMPTY:
        case FORMAT_TEXT:
        default:
            throw new DVLException("Did not recieve a data ensemble from device.");
    }
    return data;
}


































