#ifndef WRAPPER_H
#define WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Struktur sederhana untuk mengembalikan hasil ke Go
typedef struct {
    int label;
    float probability;
    int success;
} PredictionResult;

// Fungsi yang akan dipanggil dari Go
PredictionResult predict_sentiment(const char* input_text);

#ifdef __cplusplus
}
#endif

#endif
