CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall

all: strassen winograd

strassen: strassen_impl.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

winograd: winograd_impl.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f strassen winograd

.PHONY: all clean
