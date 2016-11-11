/*
 * Testing program for IO library functionality.
 */

#include <qubobus.h>
#include <io.h>

#if QUBOBUS_PROTOCOL_VERSION != 3
#error Update me with new message defs!
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int tx_fd[2]; // TO CHILD FROM PARENT
int rx_fd[2]; // TO PARENT FROM CHILD
int debug = 0;

ssize_t parent_read(void *buffer, size_t size) {
    ssize_t transferred;
    transferred = read(rx_fd[0], buffer, size);
    if (debug) {
        printf("parent read %d/%d\n", transferred, size);
        sleep(1);
    }
    return transferred;
}

ssize_t parent_write(void *buffer, size_t size) {
    ssize_t transferred;
    transferred = write(tx_fd[1], buffer, size);
    if (debug) {
        printf("parent write %d/%d\n", transferred, size);
        sleep(1);
    }
    return transferred;
}

ssize_t child_read(void *buffer, size_t size) {
    ssize_t transferred;
    transferred = read(tx_fd[0], buffer, size);
    if (debug) {
        printf("child read %d/%d\n", transferred, size);
        sleep(1);
    }
    return transferred;
}

ssize_t child_write(void *buffer, size_t size) {
    ssize_t transferred;
    transferred = write(rx_fd[1], buffer, size);
    if (debug) {
        printf("child write %d/%d\n", transferred, size);
        sleep(1);
    }
    return transferred;
}

int parent_program() {
    IO_State state_storage, *state = &state_storage;
    int error = 0;

    initialize(state, &parent_read, &parent_write, &malloc, &free, 40);

    printf("Parent connecting...\n");

    error |= connect(state);

    printf("Parent connected!\n");

    {
        int data = 1337;
        Message m = create_message(state, MT_REQUEST, 13, &data, sizeof(int));
        send_message(state, &m);
        destroy_message(state, &m);
    }

    return error;
}

int child_program() {
    IO_State state_storage, *state = &state_storage;
    int error = 0;

    initialize(state, &child_read, &child_write, &malloc, &free, 80);

    printf("Child connecting...\n");

    error |= connect(state);

    printf("Child connected!\n");

    {
        int data = 1336;
        Message m = recieve_message(state, &data, sizeof(int));

        if (m.header.message_type != MT_REQUEST)
            error = 4;
        else if (m.header.message_id != 13)
            error = 5;
        else if (data != 1337)
            error = 6;
        destroy_message(state, &m);
    }

    return error;
}

int main() { 
    int error = 0, child_pid;

    if (pipe(tx_fd) < 0) {

        printf("Unable to open transmit pipe!\n");

        error = 1;

    } else if (pipe(rx_fd) < 0) {

        printf("Unable to open recieve pipe!\n");

        close(tx_fd[0]); // CHILD READ
        close(tx_fd[1]); // PARENT WRITE

        error = 2;

    } else if ((child_pid = fork()) < 0) {

        printf("Unable to fork!\n");

        close(tx_fd[0]); // CHILD READ
        close(tx_fd[1]); // PARENT WRITE

        close(rx_fd[0]); // PARENT READ
        close(rx_fd[1]); // CHILD WRITE

        error = 3;

    } else if (child_pid > 0) {

        int child_error;

        close(tx_fd[0]); // CHILD READ
        close(rx_fd[1]); // CHILD WRITE

        error = parent_program();

        close(tx_fd[1]); // PARENT WRITE
        close(rx_fd[0]); // PARENT READ

        wait(&child_error);

        error |= child_error;

        if (!error) {
            printf("Protocol test successful!\n");
        }

    } else {

        close(tx_fd[1]); // PARENT WRITE
        close(rx_fd[0]); // PARENT READ

        error = child_program();

        close(tx_fd[0]); // CHILD READ
        close(rx_fd[1]); // CHILD WRITE

    }

    return error;
}
