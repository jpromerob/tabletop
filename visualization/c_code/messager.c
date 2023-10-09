#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <time.h>
#include <math.h>

#define CONTROLLER_IP "172.16.222.30"
#define CONTROLLER_PORT 5151
#define PLOTTER_IP "172.16.222.199"
#define PLOTTER_PORT 1987

int main() {
    int controller_socket, plotter_socket;
    struct sockaddr_in controller_addr, plotter_addr;

    // Create UDP sockets
    controller_socket = socket(AF_INET, SOCK_DGRAM, 0);
    plotter_socket = socket(AF_INET, SOCK_DGRAM, 0);

    if (controller_socket == -1 || plotter_socket == -1) {
        perror("Socket creation error");
        exit(EXIT_FAILURE);
    }

    // Initialize sockaddr_in structures for the controller and plotter
    memset(&controller_addr, 0, sizeof(controller_addr));
    controller_addr.sin_family = AF_INET;
    controller_addr.sin_port = htons(CONTROLLER_PORT);
    controller_addr.sin_addr.s_addr = inet_addr(CONTROLLER_IP);

    memset(&plotter_addr, 0, sizeof(plotter_addr));
    plotter_addr.sin_family = AF_INET;
    plotter_addr.sin_port = htons(PLOTTER_PORT);
    plotter_addr.sin_addr.s_addr = inet_addr(PLOTTER_IP);

    int step = 0;
    while (1) {
        // Generate random coordinates

        if(step == 360){
            step = 0;
        } else {
            step++;
        }

        double x_cm = 10*cos(step * M_PI / 180.0);
        double y_cm = 43+10*sin(step * M_PI / 180.0);

        // Create a message string
        char message[50];
        snprintf(message, sizeof(message), "%.2f,%.2f", x_cm, y_cm);
        message[sizeof(message) - 1] = '\0'; // Add null-terminator
        printf("%s\n", message);

        // Send the message to the controller
        sendto(controller_socket, message, strlen(message), 0,
               (struct sockaddr *)&controller_addr, sizeof(controller_addr));

        // Send the message to the plotter
        sendto(plotter_socket, message, strlen(message), 0,
               (struct sockaddr *)&plotter_addr, sizeof(plotter_addr));

        usleep(1000*10); // 10ms
    }

    // Close sockets (this part of the code may not be reached)
    close(controller_socket);
    close(plotter_socket);

    return 0;
}