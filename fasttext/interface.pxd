# fastText C++ interface
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport int32_t, int64_t
from libcpp.memory cimport shared_ptr

cdef extern from "../../src/real.h":
    ctypedef float real

cdef extern from "../../src/args.h":
    cdef cppclass Args:
        Args()

cdef extern from "../../src/dictionary.h":
    cdef cppclass Dictionary:
        Dictionary(shared_ptr[Args])

        int32_t nwords()
        int32_t nlabels()

        string getWord(int32_t)
        string getLabel(int32_t)

cdef extern from "interface.h":
    cdef cppclass FastTextModel:
        FastTextModel()
        int dim
        int ws
        int epoch
        int minCount
        int minCountLabel
        int neg
        int wordNgrams
        string lossName
        string modelName
        int bucket
        int minn
        int maxn
        int lrUpdateRate
        double t
        int verbose

        int saveVectors
        int saveLabelVectors
        string dictionary
        double dropInput
        int lrFreeze

        vector[string] getWords()
        vector[real] getVectorWrapper(string word)
        vector[double] classifierTest(string filename, int32_t k)
        vector[string] classifierPredict(string text, int32_t k)
        vector[vector[string]] classifierPredictProb(string text, int32_t k)

        # Wrapper for Dictionary class
        int32_t dictGetNWords()
        string dictGetWord(int32_t i)
        int32_t dictGetNLabels()
        string dictGetLabel(int32_t i)
        vector[real] dictGetLabelVector(int32_t i)
        vector[int64_t] dictGetWordCounts()
        vector[int64_t] dictGetLabelCounts()

    void trainWrapper(int argc, char **argvm, int silent)

    # Add 'except +' to the function declaration to let Cython safely raise an
    # appropriate Python exception instead
    void loadModelWrapper(string filename, FastTextModel& model) except +


