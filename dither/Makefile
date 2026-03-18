CC = gcc
CFLAGS = -O3 -mavx2 -mfma -march=native -Wall -Wextra
LDFLAGS = -lm -lrt

TARGET = dither_test
OBJS = main.o dither_avx.o

.PHONY: all clean test run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $(TARGET)

main.o: main.c dither_avx.h
	$(CC) $(CFLAGS) -c main.c -o main.o

dither_avx.o: dither_avx.c dither_avx.h
	$(CC) $(CFLAGS) -c dither_avx.c -o dither_avx.o

clean:
	rm -f $(OBJS) $(TARGET)

test: $(TARGET)
	./$(TARGET) --test-all

run: $(TARGET)
	./$(TARGET) --benchmark

bench: $(TARGET)
	./$(TARGET) --benchmark --vocab 32000
