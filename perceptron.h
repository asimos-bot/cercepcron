
typedef float dtype;

typedef struct {
  dtype* weights;
  unsigned int weightsLen;
} SLP;

typedef SLP SingleLayerPerceptron;

SLP* createSLP(unsigned int inputLen);

void freeSLP(SLP* slp);

int forwardSLP(SLP* slp, dtype* x);

typedef struct {
  dtype maxNumIterations;
  dtype learningRate;
  // rows is predicted, columns is actual
} SLPTrainConfig;

SLPTrainConfig* createSLPTrainConfig();

void trainSLP(SLP* slp, SLPTrainConfig* config, dtype* x, unsigned int* y, unsigned int numSamples, float trainPercentage, float** accuracy, unsigned int* accuracyLen);
