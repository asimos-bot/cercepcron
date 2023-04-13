#include "perceptron.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <omp.h>
#include <unistd.h>
#include <sys/ioctl.h>

#define WIDTH 400
#define HEIGHT 400
#define MIN(a, b) a < b ? a : b
#define CIRCLE 0
#define RECT 1

#define NUM_SAMPLES 100
#define NUM_ITERS 1000
#define LEARNING_RATE 0.9
#define ERROR 0.001

#define debug fprintf(stderr, "%s:%u\n", __FILE__, __LINE__)

#define return_defer(value) do { result = (value); goto defer; } while(0)

int toPPM(dtype* img, char* filename) {
    int result = 0;
    FILE* file = NULL;
    if(!( file = fopen(filename, "wb") )) return_defer(errno);

    fprintf(file, "P6\n%u %u\n255\n", WIDTH, HEIGHT);
    if(ferror(file)) return_defer(errno);
    for(unsigned int i = 0; i < WIDTH * HEIGHT; i++) {
        uint8_t bytes[3] = {
            !!img[i] * 255,
            !!img[i] * 255,
            !!img[i] * 255
        };
        fwrite(bytes, sizeof(bytes), 1, file);
        if(ferror(file)) return_defer(errno);
    }

defer:
    if(file) fclose(file);
    return result;
}

dtype* rect(dtype* img) {
    int centerX = rand() % WIDTH;
    int centerY = rand() % HEIGHT;
    int width = rand() % MIN(WIDTH/2, HEIGHT/2);
    int height = rand() % MIN(WIDTH/2, HEIGHT/2);
    for(int i = 0; i < HEIGHT; i++) {
        for(int j = 0; j < WIDTH; j++) {
            int distY = abs(centerY - (int)i);
            int distX = abs(centerX - (int)j);
            img[i * WIDTH + j] = distX <= width && distY <= height;
        }
    }
    return img;
}

dtype* circle(dtype* img) {
    int centerX = rand() % WIDTH;
    int centerY = rand() % HEIGHT;
    int radius = rand() % MIN(WIDTH/3, HEIGHT/3);
    radius *= radius;
    for(int i = 0; i < HEIGHT; i++) {
        for(int j = 0; j < WIDTH; j++) {
            int distY = abs(centerY - (int)i);
            int distX = abs(centerX - (int)j);
            int dist = distY * distY + distX * distX;
            img[i * WIDTH + j] = dist <= radius;
        }
    }
    return img;
}

typedef struct {
    dtype* x;
    unsigned int* y;
} Dataset;

Dataset* createDataset() {

    Dataset* dataset = malloc(sizeof(Dataset));
    memset(dataset, 0x00, sizeof(Dataset));
    dataset->x = calloc(WIDTH * HEIGHT * NUM_SAMPLES, sizeof(dtype));
    dataset->y = calloc(NUM_SAMPLES, sizeof(unsigned int));

    // circles
    unsigned int i = 0;
    for(; i < NUM_SAMPLES/2; i++) {
        rect(&dataset->x[i * WIDTH * HEIGHT]);
        dataset->y[i] = RECT;
        toPPM(&dataset->x[i * WIDTH * HEIGHT], "rect.ppm");
    }
    // rects
    for(; i < NUM_SAMPLES; i++) {
        circle(&dataset->x[i * WIDTH * HEIGHT]);
        dataset->y[i] = CIRCLE;
    }
    return dataset; 
}

void map_metric_to_plot_points(float* metric, unsigned int len, unsigned int* points, unsigned int term_width, unsigned int term_height) {

    if(!len) {
        for(unsigned int i = 0; i < term_width; i++) points[i] = term_height-1;
        return;
    }
    float scale_width = (float)term_width/len;
    float scale_height = (float)term_height;
    for(unsigned int i = 0; i < term_width; i++) {
        unsigned int metric_idx = scale_width * i;
        float value = metric[metric_idx] * scale_height;
        points[i] = term_height - ((unsigned int) value) - 1;
    }
    if(len < term_width) {
        for(unsigned int i = len; i < term_width; i++) {
            points[i] = term_height;
        }
    }
}

void ascii_plot(float* metric, unsigned int len) {
    struct winsize w;
    ioctl(0, TIOCGWINSZ, &w);

    unsigned int term_height = w.ws_row < 100 ? w.ws_row : 99;
    unsigned int term_width = w.ws_col;
    char term[term_height][term_width+1];
    memset(term, ' ', term_height * (term_width+1));

    // status line
    char status_line[term_width];
    memset(status_line, 0x00, term_width);
    sprintf(status_line, "accuracy: %f, iter: %u", len ? metric[len-1] : 0, len);
    memcpy(term, status_line, strlen(status_line));

    for(unsigned int i = 0; i < term_height; i++) term[i][term_width] = 0x00;

    unsigned int points[term_width];
    map_metric_to_plot_points(metric, len, points, term_width, term_height-1);
    for(unsigned int i = 0; i < term_width; i++) {
        if(points[i] < term_height-1) {
            term[points[i]+1][i] = '-';
        }
    }
    printf("\n");
    for(unsigned int i = 0; i < term_height; i++) {
        printf("%s", term[i]);
        if(i != term_height-1) printf("\n");
    }
    fflush(stdout);
}

int main() {
    srand(1337);

    Dataset* dataset = createDataset();

    float* accuracy = NULL;
    unsigned int accuracyLen = 0;

    SLPTrainConfig* config = createSLPTrainConfig();
    config->learningRate = LEARNING_RATE;
    config->errorThreshold = ERROR;
    config->maxNumIterations = NUM_ITERS;
    SLP* slp = createSLP(WIDTH * HEIGHT);

#if defined(_OPENMP)
    #pragma omp parallel num_threads(2)
    {
        #pragma omp single
        {
            int num_threads = omp_get_num_threads();
            if(num_threads < 2) {
                fprintf(stderr, "2 threads at least are needed!");
                exit(1);
            }
        }
        #pragma omp barrier
        int stop = 0;
        int id = omp_get_thread_num();
        if(id == 0) {
            unsigned int currentAccuracyLen = accuracyLen;
            ascii_plot(accuracy, currentAccuracyLen);
            while(!stop) {
                if(accuracyLen != currentAccuracyLen) {
                    currentAccuracyLen = accuracyLen;
                    ascii_plot(accuracy, accuracyLen);
                } 
                sleep(1);
            }
            ascii_plot(accuracy, accuracyLen);
        } else if(id == 1) {
            trainSLP(slp, config, dataset->x, dataset->y, NUM_SAMPLES, 1, &accuracy, &accuracyLen);
            stop = 1;
        }
    }
#else

    trainSLP(slp, config, dataset->x, dataset->y, NUM_SAMPLES, 1, &accuracy, &accuracyLen);
#endif
    return 0;
}
