CC = g++
FILES=facerec_video.cpp

exemplo: clean
	$(CC) `pkg-config --cflags --libs /usr/local/Cellar/opencv/2.4.13_3/lib/pkgconfig/opencv.pc` $(FILES) -o main

clean:
	- rm main
