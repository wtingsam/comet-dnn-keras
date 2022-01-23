CXX = g++
APPDIR = app
BINDIR = bin
LIBDIR = lib
SRCDIR = src
SUFFIX = .cxx
HEADER = .hxx

CXXFLAGS += -g -W -Wall -O2 -lSpectrum
CXXFLAGS += $(shell root-config --cflags)
LIBS     += $(shell $(ROOTSYS)/bin/root-config --glibs) -lMinuit -lTMVA
LIBS     += -pthread -lm -ldl -rdynamic
INCS     += -I$(shell $(ROOTSYS)/bin/root-config --incdir)
INCS     += -Isrc -Iapp -Isrc/Tracking -Isrc/Histogramming -Isrc/PreTrackStrategies 

SRCA := $(wildcard $(APPDIR)/*$(SUFFIX))
#SRCS := $(wildcard $(SRCDIR)/./$(SUFFIX))
SRCS := $(shell find src -type f -iname '*.cxx')
HEAD := $(shell find src -type f -iname '*.hxx')

OBJS = $(addprefix $(LIBDIR)/, $(notdir $(SRCS:$(SUFFIX)=.o)))
TGTS = $(addprefix $(BINDIR)/, $(notdir $(basename $(SRCA))))

.PHONY: all
all: $(TGTS) $(OBJS)

# If a file in app changed, make only that app by looking at the Library
$(BINDIR)/%: $(OBJS) $(APPDIR)/%$(SUFFIX) 
	$(CXX) $(CXXFLAGS) $(LIBS) $(INCS) $(APPDIR)/$(notdir $@)$(SUFFIX) -o $@ $(filter-out $(APPDIR)/%$(SUFFIX), $^)

# If source .cxx/.hxx changed, make all .o files in library
$(LIBDIR)/%.o: $(SRCDIR)/%$(SUFFIX) $(SRCDIR)/%$(HEADER) 
	$(CXX) -c $(CXXFLAGS) ${INCS} $< -o $@ 

# If source .cxx/.hxx changed in the second level of directory, make all .o files 
$(LIBDIR)/%.o: $(SRCDIR)/*/%$(SUFFIX) $(SRCDIR)/*/%$(HEADER) 
	$(CXX) -c $(CXXFLAGS) ${INCS} $< -o $@

.PHONY: clean
clean:	
	rm -f $(OBJS) $(TGTS)
