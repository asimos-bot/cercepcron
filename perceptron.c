#include "perceptron.h"
#include <string.h>
#include <stdlib.h>

SLP* createSLP(unsigned int inputLen) {
  SLP* slp = malloc(sizeof(SLP));
  slp->weightsLen = inputLen + 1; // +1 for bias (first weight)
  slp->weights = calloc(slp->weightsLen, sizeof(dtype));
  memset(slp->weights, 0x00, sizeof(dtype) * slp->weightsLen);
  return slp;
}

void freeSLP(SLP* slp) {
  free(slp->weights);
  free(slp);
}
int forwardSLP(SLP* slp, dtype* x) {
  dtype y = slp->weights[0];
  for(unsigned int i = 1; i < slp->weightsLen; i++) y += slp->weights[i] * x[i-1];
  return 0 <= y;
}

SLPTrainConfig* createSLPTrainConfig() {
  SLPTrainConfig* config = malloc(sizeof(SLPTrainConfig));
  memset(config, 0x00, sizeof(SLPTrainConfig));
  return config;
}

#if !defined(_OPENMP)
    #include <stdio.h>
#endif

void trainSLPIter(SLP* slp, SLPTrainConfig* config, dtype* sample, dtype sampleLabel) {
      int predictedClass = forwardSLP(slp, sample);
      int diff = sampleLabel - predictedClass;
      // bias
      slp->weights[0] += config->learningRate * diff;
      // other weights
      for(unsigned long i = 1; i < slp->weightsLen; i++) {
          slp->weights[i] += config->learningRate * diff * sample[i-1];
      }
}

void trainSLP(SLP* slp, SLPTrainConfig* config, dtype* x, unsigned int* y, unsigned int numSamples, float trainPercentage, float** accuracy, unsigned int* accuracyLen) {
    
  *accuracyLen = 0;
  unsigned int trainLen = numSamples * trainPercentage;
  unsigned int testLen = numSamples - trainLen;
  // iterations
  for(unsigned long iter = 0; iter < config->maxNumIterations; iter++) {

    // train
    for(unsigned long sample = 0; sample < testLen; sample++) {
        trainSLPIter(slp, config, &x[(slp->weightsLen - 1) * sample], y[sample]);
    }
    // test
    unsigned int correct = 0;
    for(unsigned long sample = trainLen; sample < numSamples; sample++) {
        correct += forwardSLP(slp, &x[(slp->weightsLen-1)* sample]) == y[sample];
    }
    *accuracy = realloc(*accuracy, ((*accuracyLen)+1) * sizeof(float));
    (*accuracy)[(*accuracyLen)] = (float)correct/(testLen);
#if !defined(_OPENMP)
    printf("iter: %lu, accuracy: %f\n", iter, (*accuracy)[(*accuracyLen)]); 
#endif
    (*accuracyLen)++;
  }
}
