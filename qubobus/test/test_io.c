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

ssize_t pipe_write(void *io_host, void *buffer, size_t size) {
    return write(((int*)io_host)[1], buffer, size);
}

ssize_t pipe_read(void *io_host, void *buffer, size_t size) {
    return read(((int*)io_host)[0], buffer, size);
}

int parent_program() {
    IO_State state_storage, *state = &state_storage;
    int pipefd[2] = {rx_fd[0], tx_fd[1]}, error = 0;

    state_storage = initialize(&pipefd, &pipe_read, &pipe_write, 40);

    printf("Parent connecting...\n");

    error |= connect(state);

    printf("Parent connected!\n");

    {
        struct Depth_Status depth_status = {3.14f, 2};
        Message m = create_response(&tDepthStatus, &depth_status);
        write_message(state, &m);
    }

    return error;
}

int child_program() {
    IO_State state_storage, *state = &state_storage;
    int pipefd[2] = {tx_fd[0], rx_fd[1]}, error = 0;

    state_storage = initialize(pipefd, &pipe_read, &pipe_write, 80);

    printf("Child connecting...\n");

    error |= connect(state);

    printf("Child connected!\n");

    {
        struct Depth_Status depth_status = {0.14f, 1};
        Message m;
        read_message(state, &m, &depth_status);

        if (m.header.message_type != MT_RESPONSE)
            error = 4;
        else if (m.header.message_id != tDepthStatus.id)
            error = 5;
        else if (depth_status.warning_level != 2)
            error = 6;
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
