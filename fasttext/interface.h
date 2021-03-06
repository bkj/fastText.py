#ifndef FASTTEXT_INTERFACE_H
#define FASTTEXT_INTERAFCE_H

#include <string>
#include <vector>
#include <memory>

#include "../../src/real.h"
#include "../../src/args.h"
#include "../../src/dictionary.h"
#include "../../src/matrix.h"
#include "../../src/model.h"

using namespace fasttext;

class FastTextModel {
    private:
        std::vector<std::string> _words;
        std::shared_ptr<Dictionary> _dict;
        std::shared_ptr<Matrix> _input_matrix;
        std::shared_ptr<Matrix> _output_matrix;
        std::shared_ptr<Model> _model;

    public:
        FastTextModel();
        int dim;
        int ws;
        int epoch;
        int minCount;
        int minCountLabel;
        int neg;
        int wordNgrams;
        std::string lossName;
        std::string modelName;
        int bucket;
        int minn;
        int maxn;
        double lr;
        int lrUpdateRate;
        double t;
        int verbose;
        
        int saveVectors;
        int saveLabelVectors;
        std::string dictionary;
        double dropInput;
        int lrFreeze;

        std::vector<std::string> getWords();
        std::vector<real> getVectorWrapper(std::string word);
        std::vector<double> classifierTest(std::string filename, int32_t k);
        std::vector<std::string> classifierPredict(std::string text, int32_t k);
        std::vector<std::vector<std::string>> classifierPredictProb(std::string text,
                int32_t k);

        void addWord(std::string word);
        void setArgs(std::shared_ptr<Args> args);
        void setDictionary(std::shared_ptr<Dictionary> dict);
        void setMatrix(std::shared_ptr<Matrix> input,
                std::shared_ptr<Matrix> output);
        void setModel(std::shared_ptr<Model> model);

        /* wrapper for Dictionary class */
        int32_t dictGetNWords();
        std::string dictGetWord(int32_t i);
        int32_t dictGetNLabels();
        std::string dictGetLabel(int32_t i);
        std::vector<real> dictGetLabelVector(int32_t i);
        std::vector<int64_t> dictGetWordCounts();
        std::vector<int64_t> dictGetLabelCounts();
};

void trainWrapper(int argc, char **argv, int silent);
void loadModelWrapper(std::string filename, FastTextModel& model);

#endif

