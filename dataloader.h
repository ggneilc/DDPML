#ifndef __DATALOADER_H__
#define __DATALOADER_H__
#include "matrix.h"

typedef struct dataloader {
    int features;
    int samples;
    char* labels;
    matrix X;
    vector y;
} dataloader;

dataloader* load_data(char* filepath);


#endif // __DATALOADER_H__