#include <libcaercpp/devices/dvxplorer.hpp>

#include <atomic>
#include <csignal>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <arpa/inet.h>
using namespace std;




#define IP_VISUAL "172.16.222.199"
#define PORT_VISUAL 3330
#define IP_SPINN "172.16.223.2"
#define PORT_SPINN 3333
#define NUM_EVENTS 64 // Number of events to send


#define MAX_PIXIX 2

struct lutmap {
        uint8_t np;
        int x[MAX_PIXIX];
        int y[MAX_PIXIX];
};

/* Function: get_new_lut
 * ---------------------
 * Fills a LUT, an array of type lutmap, with 'meaningful' starting data
 *
 * width: expected image width
 * height: expected image height
 * lut: pointer towards allocated LUT
 * empty: flag indicating if LUT should be empty or not
 *      a 'non-empty' LUT maps input/output events 1-to-1
 *      this means (x,y) --> (x,y)
 *
 */
void get_new_lut(int width, int height, lutmap lut[], bool empty){    

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {   
            if(empty){         
                lut[x*height+y].np = 0;
                lut[x*height+y].x[0] = -1;            
                lut[x*height+y].y[0] = -1;   
            } else {
                lut[x*height+y].np = 1;
                lut[x*height+y].x[0] = x;            
                lut[x*height+y].y[0] = y;  
            }       
            lut[x*height+y].x[1] = -1;      
            lut[x*height+y].y[1] = -1;
        }
    }
}

/* Function: load_undistortion_lut
 * ---------------------
 * Fills a LUT with content from a *.csv file
 *
 * fname: name of the *.csv file
 * width: expected image width
 * height: expected image height
 * lut: pointer towards allocated LUT
 *
 */
void load_undistortion_lut(const std::string & fname, lutmap lut[]){

    
    // File pointer
    std::ifstream infile(fname);  

    vector<string> row;
    string line, val, temp;  

    while (std::getline(infile, line))
    {
        stringstream s(line);
  
        int col_idx = 0;
        int lut_idx;
        uint8_t np = 0;
        while (getline(s, val, ',')) 
        {
            switch(col_idx) {
                case 0:
                    lut_idx = stoi(val);
                    np = 0;
                    break;
                case 1:
                    lut[lut_idx].x[0]= stoi(val);
                    if (lut[lut_idx].x[0] >= 0){
                        np += 1;
                    }
                    break;
                case 2:
                    lut[lut_idx].y[0]= stoi(val);
                    lut[lut_idx].np= np;
                    break;
                case 3:
                    lut[lut_idx].x[1]= stoi(val);
                    if (lut[lut_idx].x[1] >= 0){
                        np += 1;
                    }
                    break;
                case 4:
                    lut[lut_idx].y[1]= stoi(val);
                    lut[lut_idx].np= np;
                    break;
                case 5:
                    break;
            }
            col_idx += 1;
        }
    }
    
    
}

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

	printf("%s --- ID: %d, DVS X: %d, DVS Y: %d, Firmware: %d, Logic: %d.\n", info.deviceString, info.deviceID,
		info.dvsSizeX, info.dvsSizeY, info.firmwareVersion, info.logicVersion);

	// Send the default configuration before using the device.
	// No configuration is sent automatically!
	// dvXplorerConfigSet(cdh, DVX_USB, DVX_USB_EARLY_PACKET_DELAY,8) // in 125µs time-slices (defaults to 1ms)
	handle.sendDefaultConfig();
	// handle.configSet(DVX_USB, DVX_USB_EARLY_PACKET_DELAY, 8);

	// Now let's get start getting some data from the device. We just loop in blocking mode,
	// no notification needed regarding new events. The shutdown notification, for example if
	// the device is disconnected, should be listened to.
	handle.dataStart(nullptr, nullptr, nullptr, &usbShutdownHandler, nullptr);

	// Let's turn on blocking data-get mode to avoid wasting resources.
	handle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);
	handle.configSet(CAER_HOST_CONFIG_PACKETS,CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_INTERVAL, 100);
	handle.configSet(CAER_HOST_CONFIG_PACKETS,CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_PACKET_SIZE, 1);
	handle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BUFFER_SIZE, 0);
	// handle.configSet();



    uint16_t max_dim = std::max(info.dvsSizeX,info.dvsSizeY);

    /* Create the main LUT, i.e. an array of type 'lutmap'*/
    lutmap * lut = (lutmap*)malloc(max_dim*max_dim*sizeof(lutmap));

	std::string undistortion_filename = "cam_lut_homography.csv";

    /* Fill the main LUT with meaningful data: with or without undistortion */
    if(undistortion_filename.length() > 0){        
        load_undistortion_lut(undistortion_filename, lut);
    } else {
        get_new_lut(info.dvsSizeX, info.dvsSizeY, lut, false);  
    }

    int pShift = 15;
    int yShift = 0;
    int xShift = 16;
    unsigned int noTimestamp = 0x80000000;



    int sock_visual;
    struct sockaddr_in server_visual;
    sock_visual = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_visual == -1) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }
    server_visual.sin_family = AF_INET;
    server_visual.sin_port = htons(PORT_VISUAL);
    server_visual.sin_addr.s_addr = inet_addr(IP_VISUAL);



    int sock_spinn;
    struct sockaddr_in server_spinn;
    sock_spinn = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_spinn == -1) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }
    server_spinn.sin_family = AF_INET;
    server_spinn.sin_port = htons(PORT_SPINN);
    server_spinn.sin_addr.s_addr = inet_addr(IP_SPINN);

    unsigned int *eventArray = (unsigned int *)malloc(NUM_EVENTS * sizeof(unsigned int));
    if (eventArray == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }
	unsigned int *empty = (unsigned int *)malloc(NUM_EVENTS * sizeof(unsigned int));
    if (empty == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

	for (int i = 0; i < NUM_EVENTS; i++) {
		eventArray[i] = 0;
		empty[i] = 0;
	}


	int udp_ev_counter = 0;
	while (!globalShutdown.load(memory_order_relaxed)) {
		std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = handle.dataGet();
		if (packetContainer == nullptr) {
			continue; // Skip if nothing there.
		}

		for (auto &packet : *packetContainer) {
			if (packet == nullptr) {
				// printf("Packet is empty (not present).\n");
				continue; // Skip if nothing there.
			}

			if (packet->getEventType() == POLARITY_EVENT) {
				std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity
					= std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

				// Get full timestamp and addresses of first event.

				for (int i = 0; i < packet->getEventNumber(); i++){
					const libcaer::events::PolarityEvent &firstEvent = (*polarity)[i];

                    for(int pixix = 0; pixix < lut[firstEvent.getX()*info.dvsSizeY+firstEvent.getY()].np; pixix++){
                        uint16_t x = lut[firstEvent.getX()*info.dvsSizeY+firstEvent.getY()].x[pixix];
                        uint16_t y = lut[firstEvent.getX()*info.dvsSizeY+firstEvent.getY()].y[pixix];
                        bool pol = true;
                        eventArray[udp_ev_counter] = noTimestamp + (pol << pShift) + (y << yShift) + (x << xShift);
                        // printf("...\n");	
                        udp_ev_counter++;
                        if (udp_ev_counter == NUM_EVENTS){
                            // sendto(sock_visual, eventArray, NUM_EVENTS * sizeof(unsigned int), 0, (struct sockaddr *)&server_visual, sizeof(server_visual));
                            sendto(sock_spinn, eventArray, NUM_EVENTS * sizeof(unsigned int), 0, (struct sockaddr *)&server_spinn, sizeof(server_spinn));
                            memcpy(eventArray, empty, NUM_EVENTS * sizeof(unsigned int));
                            udp_ev_counter = 0;
                        }
                    }
                    
                    
					// printf("(%u,%u)\n", x, y);	
					
				}

				// int b = x << 2;

			}
		}
	}

	handle.dataStop();


	printf("Shutdown successful.\n");

	return (EXIT_SUCCESS);
}
