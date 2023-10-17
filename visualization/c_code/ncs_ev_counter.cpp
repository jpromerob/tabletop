#include <libcaercpp/devices/dvxplorer.hpp>

#include <atomic>
#include <csignal>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <iostream>
#include <chrono>
#include <thread>


#define ACC_TIME  1000 // 1000 ms
using namespace std;

static atomic_bool globalShutdown(false);

static void globalShutdownSignalHandler(int signal) {
	// Simply set the running flag to false on SIGTERM and SIGINT (CTRL+C) for global shutdown.
	if (signal == SIGTERM || signal == SIGINT) {
		globalShutdown.store(true);
	}
}

static void usbShutdownHandler(void *ptr) {
	(void) (ptr); // UNUSED.

	globalShutdown.store(true);
}
int main(void) {


	struct sigaction shutdownAction;

	shutdownAction.sa_handler = &globalShutdownSignalHandler;
	shutdownAction.sa_flags   = 0;
	sigemptyset(&shutdownAction.sa_mask);
	sigaddset(&shutdownAction.sa_mask, SIGTERM);
	sigaddset(&shutdownAction.sa_mask, SIGINT);

	if (sigaction(SIGTERM, &shutdownAction, NULL) == -1) {
		libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
			"Failed to set signal handler for SIGTERM. Error: %d.", errno);
		return (EXIT_FAILURE);
	}

	if (sigaction(SIGINT, &shutdownAction, NULL) == -1) {
		libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
			"Failed to set signal handler for SIGINT. Error: %d.", errno);
		return (EXIT_FAILURE);
	}

	// Open a DVS, give it a device ID of 1, and don't care about USB bus or SN restrictions.
	auto handle = libcaer::devices::dvXplorer(1);

	// Let's take a look at the information we have on the device.
	auto info = handle.infoGet();

	printf("Camera ID: %s\n", info.deviceString);
	printf("Device ID: %d\n", info.deviceID);
	printf("Resolution: %d x %d\n", info.dvsSizeX, info.dvsSizeY);

	// Send the default configuration before using the device.
	// No configuration is sent automatically!
	handle.sendDefaultConfig();

	// Now let's get start getting some data from the device. We just loop in blocking mode,
	// no notification needed regarding new events. The shutdown notification, for example if
	// the device is disconnected, should be listened to.
	handle.dataStart(nullptr, nullptr, nullptr, &usbShutdownHandler, nullptr);

	// Let's turn on blocking data-get mode to avoid wasting resources.
	handle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);



	while (!globalShutdown.load(memory_order_relaxed)) {

		auto start_time = std::chrono::high_resolution_clock::now(); 
		int ev_counter = 0;
		while(true){

			std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = handle.dataGet();
			if (packetContainer == nullptr) {
				continue; // Skip if nothing there.
			}

			for (auto &packet : *packetContainer) {
				if (packet == nullptr) {
					continue; // Skip if nothing there.
				}

				if (packet->getEventType() == POLARITY_EVENT) {
					std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity
						= std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);
					int new_evs = packet->getEventNumber();
					ev_counter += new_evs;
					

				}
			}
			auto current_time = std::chrono::high_resolution_clock::now();
			auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);

			if (elapsed_time.count() >= ACC_TIME) {
				break; // Exit the loop after 10 ms have elapsed
        	}

		}
		if(ev_counter > 1000000){
			printf("%d Mev/s\n", ev_counter/1000000);
		} else if(ev_counter > 1000) {
			printf("%d kev/s\n", ev_counter/1000);
		} else {
			printf("%d ev/s\n", ev_counter/1);
		}
		

	}

	handle.dataStop();


	printf("Shutdown successful.\n");

	return 0;
}
