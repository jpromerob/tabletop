#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#define BUFLEN 512
#define PORT 7373

int main() {
    struct sockaddr_in si_me, si_other;
    int s, i, slen = sizeof(si_other);
    char buf[BUFLEN];

    if ((s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1) {
        perror("socket");
        exit(1);
    }

    memset((char *)&si_me, 0, sizeof(si_me));
    si_me.sin_family = AF_INET;
    si_me.sin_port = htons(PORT);
    si_me.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(s, (struct sockaddr *)&si_me, sizeof(si_me)) == -1) {
        perror("bind");
        exit(1);
    }

    while (1) {
        if (recvfrom(s, buf, BUFLEN, 0, (struct sockaddr *)&si_other, &slen) == -1) {
            perror("recvfrom");
            exit(1);
        }

        // Decode message into two floats
        float x, y;
        sscanf(buf, "%f,%f", &x, &y);

        printf("Received message: x=%.3f, y=%.3f\n", x, y);
    }

    close(s);
    return 0;
}