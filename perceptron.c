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

#define SLPTrainConfigTotal(config) (config->confusionMatrix[0][0] + config->confusionMatrix[1][1] + config->confusionMatrix[1][0] + config->confusionMatrix[0][1])
#define SLPTrainConfigCorrect(config) (config->confusionMatrix[0][0] + config->confusionMatrix[1][1])
#define SLPTrainConfigIncorrect(config) (config->confusionMatrix[1][0] + config->confusionMatrix[0][1])

SLPTrainConfig* createSLPTrainConfig() {
  SLPTrainConfig* config = malloc(sizeof(SLPTrainConfig));
  memset(config, 0x00, sizeof(SLPTrainConfig));
  return config;
}

float SLPAccuracy(SLPTrainConfig* config) {
  return (float)SLPTrainConfigCorrect(config)/SLPTrainConfigTotal(config);
}

#if !defined(_OPENMP)
    #include <stdio.h>
#endif

void trainSLP(SLP* slp, SLPTrainConfig* config, dtype* x, unsigned int* y, unsigned int numSamples, float trainPercentage, float** accuracy, unsigned int* accuracyLen) {
  unsigned trainLen = trainPercentage * numSamples;
  *accuracyLen = 0;
  // iterations
  for(unsigned long iter = 0; iter < config->maxNumIterations; iter++) {

    // go through each sample
    for(unsigned long sample = 0; sample < trainLen; sample++) {
      dtype* currentInput = &x[(slp->weightsLen - 1) * sample];
      int currentLabel = y[sample];
      int predictedClass = forwardSLP(slp, currentInput);
      int diff = currentLabel - predictedClass;
      // bias
      slp->weights[0] += config->learningRate * diff;
      // other weights
      for(unsigned long i = 1; i < slp->weightsLen; i++) {
          slp->weights[i] += config->learningRate * diff * currentInput[i-1];
      }
      config->confusionMatrix[currentLabel][predictedClass]++;
    }
    // update accuracy history
    *accuracy = realloc(*accuracy, ((*accuracyLen)+1) * sizeof(float));
    (*accuracy)[(*accuracyLen)] = SLPAccuracy(config);
#if !defined(_OPENMP)
    printf("iter: %lu, accuracy: %f, total: %u, correct: %u, train_len %u, bias: %f\n", iter, (*accuracy)[(*accuracyLen)], SLPTrainConfigTotal(config), SLPTrainConfigCorrect(config), trainLen, slp->weights[0]); 
#endif
      (*accuracyLen)++;
  }
}
