/*
 * Testing program for IO library functionality.
 */

#include <qubobus.h>
#include <io.h>

#if QUBOBUS_PROTOCOL_VERSION != 2
#error Update me with new message defs!
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int tx_fd[2]; // TO CHILD FROM PARENT
int rx_fd[2]; // TO PARENT FROM CHILD

ssize_t parent_read(void *buffer, size_t size) {
    ssize_t transferred;
    transferred = read(rx_fd[0], buffer, size);
    printf("parent read %d/%d\n", transferred, size);
    return transferred;
}

ssize_t parent_write(void *buffer, size_t size) {
    ssize_t transferred;
    transferred = write(tx_fd[1], buffer, size);
    printf("parent write %d/%d\n", transferred, size);
    return transferred;
}

ssize_t child_read(void *buffer, size_t size) {
    ssize_t transferred;
    transferred = read(tx_fd[0], buffer, size);
    printf("child read %d/%d\n", transferred, size);
    return transferred;
}

ssize_t child_write(void *buffer, size_t size) {
    ssize_t transferred;
    transferred = write(rx_fd[1], buffer, size);
    printf("child write %d/%d\n", transferred, size);
    return transferred;
}

int main() { 
    int success = 1;
    IO_State state;


    pipe(tx_fd);
    pipe(rx_fd);
    switch (fork()) {
        case 0: //CHILD
            close(tx_fd[1]); // PARENT WRITE
            close(rx_fd[0]); // PARENT READ
            printf("Initializing child\n");
            initialize(&state, &child_read, &child_write, &malloc, &free, 80);

            printf("Connecting in child\n");
            success = !connect(&state);

            printf("Child had success: %d\n", success);

            break;
        default: //PARENT
            close(tx_fd[0]); // CHILD READ
            close(rx_fd[1]); // CHILD WRITE
            printf("Initializing parent\n");
            initialize(&state, &parent_read, &parent_write, &malloc, &free, 40);

            printf("Connecting in parent\n");
            success = !connect(&state);

            printf("Parent had success: %d\n", success);

            break;
    }

    if (success) {
        printf("No unrecoverable errors found!\n");
    }
    return !success;
}
