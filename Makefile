CXX = g++
PROJECT = grab-face
OBJ = get_face.o

LDFLAGS = $(shell pkg-config --libs opencv)
CUR_DIR = $(shell pwd)
SRC = $(CUR_DIR)/src
VPATH = $(SRC)
BUILD = $(CUR_DIR)/build
DATA_DIR = $(CUR_DIR)/data

.PHONY:all clean

all: $(DATA_DIR) $(BUILD) $(PROJECT)

$(PROJECT) : $(OBJ)
	$(CXX) $(addprefix $(BUILD)/, $^) -o $@ $(LDFLAGS)

%.o : %.cpp
	$(CXX) -c $< -o $(BUILD)/$@	
clean:
	rm -rf $(BUILD)/*.o $(BUILD) $(DATA_DIR)
	rm $(PROJECT)

$(BUILD):
	-mkdir -p $@
$(DATA_DIR):
	-mkdir -p $@
