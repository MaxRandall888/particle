CXX = g++
CXXFLAGS = -std=c++20 -O0 -g -Wall -Wextra -fopenmp

SOURCES = src/main.cpp src/octree.cpp src/physics.cpp
HEADERS = src/particle.hpp src/octree.hpp src/physics.hpp

simulation: $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@

.PHONY: test
test: tests/unit_tests
	./tests/unit_tests

tests/unit_tests: tests/unit_tests.cpp src/octree.cpp src/physics.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -I src tests/unit_tests.cpp src/octree.cpp src/physics.cpp -o $@
