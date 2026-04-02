#include "wrapper.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Struktur Data FastText
struct FastTextModel {
  std::unordered_map<std::string, std::vector<float>> vectors;
  int vector_size = 0;
};

// Fungsi internal untuk memuat binary FastText
FastTextModel load_fasttext_binary(const std::string &path) {
  FastTextModel model;
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return model;
  }

  std::string header;
  std::getline(file, header);
  std::stringstream ss(header);
  int num_words;
  if (!(ss >> num_words >> model.vector_size)) {
    return model;
  }

  model.vectors.reserve(num_words);
  for (int i = 0; i < num_words; ++i) {
    std::string word;
    char ch;
    while (file.get(ch) && ch != ' ') {
      word += ch;
    }

    std::vector<float> vec(model.vector_size);
    file.read(reinterpret_cast<char *>(vec.data()),
              model.vector_size * sizeof(float));
    model.vectors[word] = std::move(vec);

    while (file.peek() == '\n' || file.peek() == '\r' || file.peek() == ' ') {
      file.ignore();
    }
  }
  return model;
}

// Implementasi Fungsi Wrapper untuk Go
PredictionResult predict_sentiment(const char *input_text) {
  PredictionResult res = {0, 0.0f, 0};
  try {
    // 1. Singleton Initialization: Model hanya di-load satu kali
    static FastTextModel ft_model =
        load_fasttext_binary("models/maxim_fasttext.bin");
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SentimentInference");
    static Ort::SessionOptions session_options = []() {
      Ort::SessionOptions options;

      // Mengambil jumlah thread/core yang tersedia di OS secara otomatis
      unsigned int num_cpus = std::thread::hardware_concurrency();
      if (num_cpus == 0)
        num_cpus = 1; // Fallback jika tidak terdeteksi

      // Atur thread untuk operasi di dalam satu node (IntraOp)
      options.SetIntraOpNumThreads(num_cpus);

      // Mematikan paksa pengaturan affinity yang menyebabkan error tersebut
      // "1" berarti menggunakan allocator lingkungan (mencegah konflik
      // affinity)
      options.AddConfigEntry("session.use_env_allocators", "1");

      // Opsional: Optimasi graph level
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

      return options;
    }();

// Sesuaikan dengan nama file ONNX Anda
#ifdef _WIN32
    static Ort::Session session(env, L"models/maxim-sentiment-models.onnx",
                                session_options);
#else
    static Ort::Session session(env, "models/maxim-sentiment-models.onnx",
                                session_options);
#endif

    if (ft_model.vectors.empty())
      return res;

    // 2. Preprocessing
    std::string text(input_text);
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);
    text = std::regex_replace(text, std::regex("[^a-z0-9\\s]"), "");

    // 3. Tokenizing & Mean Pooling
    std::stringstream ss(text);
    std::string word;
    std::vector<float> mean_vec(ft_model.vector_size, 0.0f);
    int count = 0;

    while (ss >> word) {
      auto it = ft_model.vectors.find(word);
      if (it != ft_model.vectors.end()) {
        const auto &v = it->second;
        for (int i = 0; i < ft_model.vector_size; ++i)
          mean_vec[i] += v[i];
        count++;
      }
    }

    if (count > 1) {
      for (float &val : mean_vec)
        val /= count;
    }

    // 4. ONNX Inference
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char *input_names[] = {input_name_ptr.get()};
    const char *output_names[] = {output_name_ptr.get()};

    std::vector<int64_t> input_shape = {1, 1, (int64_t)ft_model.vector_size};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, mean_vec.data(), mean_vec.size(), input_shape.data(),
        input_shape.size());

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names,
                                      &input_tensor, 1, output_names, 1);
    float *raw_outputs = output_tensors.front().GetTensorMutableData<float>();

    auto shape_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    size_t num_classes = shape_info.GetShape().back();

    // 5. Softmax & Argmax
    float max_raw = *std::max_element(raw_outputs, raw_outputs + num_classes);
    float sum_exp = 0.0f;
    std::vector<float> probs(num_classes);

    for (size_t i = 0; i < num_classes; ++i) {
      probs[i] = std::exp(raw_outputs[i] - max_raw);
      sum_exp += probs[i];
    }

    int label = 0;
    float max_prob = 0.0f;
    for (size_t i = 0; i < num_classes; ++i) {
      probs[i] /= sum_exp;
      if (probs[i] > max_prob) {
        max_prob = probs[i];
        label = (int)i;
      }
    }

    // Sukses
    res.label = label;
    res.probability = max_prob;
    res.success = 1;
  } catch (const std::exception &e) {
    std::cerr << "Error in C++ Inference: " << e.what() << std::endl;
    res.success = 0;
  } catch (...) {
    std::cerr << "Unknown error in C++ Inference" << std::endl;
  }

  return res;
}
