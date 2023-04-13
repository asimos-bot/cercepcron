
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
  dtype errorThreshold;
  dtype maxNumIterations;
  dtype learningRate;
  // rows is predicted, columns is actual
  unsigned int confusionMatrix[2][2];
} SLPTrainConfig;

#define SLPTrainConfigTotal(config) (config->confusionMatrix[0][0] + config->confusionMatrix[1][1] + config->confusionMatrix[1][0] + config->confusionMatrix[0][1])
#define SLPTrainConfigCorrect(config) (config->confusionMatrix[0][0] + config->confusionMatrix[1][1])
#define SLPTrainConfigIncorrect(config) (config->confusionMatrix[1][0] + config->confusionMatrix[0][1])

SLPTrainConfig* createSLPTrainConfig();

float SLPAccuracy(SLPTrainConfig* config);

void trainSLP(SLP* slp, SLPTrainConfig* config, dtype* x, unsigned int* y, unsigned int numSamples, float trainPercentage, float** accuracy, unsigned int* accuracyLen);
