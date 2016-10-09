
INCDIR = include/
SRCDIR = src/
OBJDIR = obj/
BINDIR = bin/
TARGETS = $(BINDIR)test

all: $(TARGETS)

test: $(BINDIR)test
	$(BINDIR)test

$(OBJDIR)%.o: $(SRCDIR)%.c $(INCDIR)* $(OBJDIR)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@ -I$(INCDIR)

$(BINDIR)%: $(OBJDIR)%.o $(BINDIR)
	$(CC) $< -o $@

$(OBJDIR):
	mkdir $(OBJDIR)

$(BINDIR):
	mkdir $(BINDIR)

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: clean all test
