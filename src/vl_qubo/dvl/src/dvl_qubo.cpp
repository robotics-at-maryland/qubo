#include "dvl_qubo.h"
//written by Jeremy Weed


/**
 * See the header file for actual function/object descriptions
 */
DvlQuboNode::DvlQuboNode(std::shared_ptr<ros::NodeHandle> n,
    int rate, std::string name, std::string device) : QuboNode(n){

	this->name = name;

	//inits a publisher on this node
	dvlPub = n->advertise<ram_msgs::DVL_qubo>("qubo/" + name, 1000);

    //creates a refresh rate
    loop_rate.reset(new ros::Rate(rate));

	//creates/opens the DVL
	//Baud rate is currently a complete guess
	dvl.reset(new DVL(device, DVL::k115200));
	ROS_DEBUG("Device location: %s", device.c_str());

	// attempt to open the DVL, and error if it doesn't work
	try{
		dvl->openDevice();
	}catch(DVLException& ex){
		ROS_ERROR("%s", ex.what());
        dvl->closeDevice();
		return;
	}

	if(!dvl->isOpen()){
		ROS_ERROR("DVL \"%s\" didn't open succsesfully", device.c_str());
		return;
	}
    try{
        setup_dvl();
    }catch(DVLException& ex){
        ROS_ERROR("Unable to setup device");
        ROS_ERROR("%s", ex.what());
    }

    //checks the parameter server to see if we have water data,
    //sets them if we don't
    // if(!n.getParam("salinity", live_cond.salinity)){
    //     live_cond.salinity = 0;
    //     n.setParam("salinity" salinity);
    // }
    // if(!n.getParam("temperature", live_cond.temperature)){
    //     live_cond.temperature = 0;
    //     n.setParam("temperature", live_cond.temperature);
    // }


	//INSERT CONFIG HERE~~~~~~

}

DvlQuboNode::~DvlQuboNode(){
    try{
        dvl->disableMeasurement();
        dvl->closeDevice();
    }catch(DVLException& ex){
        ROS_ERROR("Unable to disable measurement and close device");
        ROS_ERROR("%s", ex.what());
    }
}

void DvlQuboNode::update(){
	static int attempts = 0;
	ram_msgs::DVL_qubo msg;

	if(!dvl->isOpen()){
		try{
			setup_dvl();
            ROS_DEBUG("DVL succsesfully setup in update method");
		}catch(DVLException& ex){
			ROS_ERROR("Attempt %i to connect to the DVL failed", attempts++);
			ROS_ERROR("%s", ex.what());
			if(attempts > DvlQuboNode::MAX_CONNECTION_ATTEMPTS){
				ROS_ERROR("Failed to find DVL, exiting node.");
				exit(-1);
			}
		}
        return;
	}
	attempts = 0;
	ROS_DEBUG("Beginning to read data from the DVL");


	try{
		sensor_data = dvl->getDVLData();
        ROS_DEBUG("DVL data got succsesfully");
	}catch(DVLException& ex){
		ROS_WARN("%s", ex.what());
		dvl->closeDevice();
		return;
	}

	//begin constructing the ros msg
	//Currently assuming the data is formatted similar to the
	//messages given to us by ROS - might be wrong
	//if this returns weird data, check the dvl documentation to
	//figure out whats not right
	switch(sensor_data.transform){
		case DVL::BEAM_COORD:
			ROS_ERROR("BEAM_COORD is not supported by this ros msg");
			break;
		case DVL::INST_COORD:
			msg.wi_x_axis = sensor_data.water_vel[0];
			msg.wi_y_axis = sensor_data.water_vel[1];
			msg.wi_z_axis = sensor_data.water_vel[2];
			msg.wi_error  = sensor_data.water_vel[3];

			msg.bi_x_axis = sensor_data.bottom_vel[0];
			msg.bi_y_axis = sensor_data.bottom_vel[1];
			msg.bi_z_axis = sensor_data.bottom_vel[2];
			msg.bi_error  = sensor_data.bottom_vel[3];
			break;

		case DVL::SHIP_COORD:
			msg.ws_transverse = sensor_data.water_vel[0];
			msg.ws_longitudinal = sensor_data.water_vel[1];
			msg.ws_normal = sensor_data.water_vel[2];

			msg.bs_transverse = sensor_data.bottom_vel[0];
			msg.bs_longitudinal = sensor_data.bottom_vel[1];
			msg.bs_normal = sensor_data.bottom_vel[2];
			break;

		case DVL::EARTH_COORD:
			msg.we_east = sensor_data.water_vel[0];
			msg.we_north = sensor_data.water_vel[1];
			msg.we_upwards = sensor_data.water_vel[2];

			msg.be_east = sensor_data.bottom_vel[0];
			msg.be_north = sensor_data.bottom_vel[1];
			msg.be_upwards = sensor_data.bottom_vel[2];
			break;
        default:
            ROS_ERROR("Invalid BEAM_COORD");
	}
	dvlPub.publish(msg);
}

void DvlQuboNode::setup_dvl(){
    //attempt to load factory settings, because we have no idea what they
    //should be
    DVL::SystemConfig config;
    DVL::DataConfig output = {};
    config.speed = DVL::k115200;
    config.parity = DVL::Parity::NO_PARITY;
    config.two_stopbits=false;
    config.auto_ensemble_cycling=true;
    config.auto_ping_cycling=true;
    config.binary_data_output=true;
    config.serial_output=true;
    config.turnkey=true;
    config.recorder_enable=false;
    output.output_type = DVL::DataOutput::ALL_DATA;
    //output.output_type = DVL::DataOutput::PARTIAL_DATA;
    //output.output_type = DVL::DataOutput::MINIMUM_DATA;
    //output.output_type = DVL::DataOutput::TEXT_DATA;
    output.profile_output[DVL::VELOCITY] = true,
    output.profile_output[DVL::CORRELATION] = true,
    output.profile_output[DVL::ECHO_INTENSITY] = true,
    output.profile_output[DVL::PERCENT_GOOD] = true,
    output.profile_output[DVL::STATUS] = true;
    dvl->loadUserSettings();
    dvl->enableMeasurement();
    dvl->setSystemConfiguration(config);
    dvl->setDataTransferConfiguration(output);
    ROS_DEBUG("DVL succsesfully setup");
    ROS_DEBUG("DVL INFO: \n%s", dvl->getSystemInfo().c_str());
}
