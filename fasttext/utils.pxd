# Wrapper for fastText Utils
cdef extern from '../../src/utils.h' namespace 'utils':
    void initTables()
    void freeTables()

    float log(float)
    float sigmoid(float)
