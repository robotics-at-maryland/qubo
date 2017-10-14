# Directory structure information
INCDIR = include/
SRCDIR = src/
OBJDIR = obj/
BINDIR = bin/

# List of object targets needed in building other modules
OBJECTS = $(addprefix $(OBJDIR), io.o protocol.o embedded.o safety.o battery.o power.o thruster.o pneumatics.o depth.o debug.o)

# List of executable targets needed
TARGETS = $(addprefix $(BINDIR), test_defs test_io)

# Rule to make all external targets
all: $(OBJECTS) $(TARGETS)

# Rule to make just include objects for external modules
include: $(OBJECTS)

# Rule to run the tester program
test: $(TARGETS)
	$(BINDIR)test_defs
	$(BINDIR)test_io

# Rule to make any object
$(OBJDIR)%.o: $(SRCDIR)%.c $(INCDIR)* $(OBJDIR) 
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@ -I$(INCDIR)

# Rule to make any executable from a single object file
$(BINDIR)%: $(OBJDIR)%.o $(BINDIR) $(OBJECTS)
	$(CC) $< -o $@  $(OBJECTS)

# Rule to create a directory for object files
$(OBJDIR):
	mkdir $(OBJDIR)

# Rule to create a directory for binary files
$(BINDIR):
	mkdir $(BINDIR)

# Rule to clean the object and binary files
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# List of targets without a backing file
.PHONY: clean all include test
