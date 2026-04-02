# BlazeApp

<a href="https://guthib">
<img src="https://github.com/Nuvantim/Maxim_sentiment_model/blob/main/image/mascot.png" width=80%>
</a>

This project is an advancement in deploying machine learning models using ONNX and FastText. The application is developed in Go as the backend, which executes C++ libraries to perform sentiment prediction based on the trained model.

The development of this project was motivated by several challenges encountered when using Python, such as dependency bloat and high memory consumption during runtime. Therefore, this project was designed as an alternative solution that offers high performance, efficiency, and lower memory usage.

> This system was developed on a Linux operating system. It is recommended to use a compatible Linux distribution. If you are using another operating system, please download the appropriate ONNX library from [the official source.](https://github.com/microsoft/onnxruntime/releases)
----

## Concept Diagram

<a href="https://guthib">
<img src="https://github.com/Nuvantim/Maxim_sentiment_model/blob/main/image/flow.png">
</a>

----

## 🛠 1. Prerequisites & Installation
### 1.1 System Components
Install essential build tools and Python utilities:

```bash
sudo apt update && sudo apt install -y \
build-essential \
python3 python3-pip python3-venv \
p7zip-full
```

### 1.2 Golang Setup
Download and install the Go runtime:

```bash
wget https://go.dev/dl/go1.26.1.linux-amd64.tar.gz && \
sudo tar -C /usr/local -xzf go1.26.1.linux-amd64.tar.gz && \
export PATH=$PATH:/usr/local/go/bin
```

### 1.3 Install Docker Engine
Install Docker to containerize your deployment.

> Follow the installation instructions below (sourced from [the Official Docker Documentation](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)):
----

## 📦 2. Model Acquisition (HuggingFace)
Use a virtual environment to securely download the required sentiment analysis models.

### 2.1 Setup Python Environment

```bash
python3 -m venv example && source example/bin/activate && \
pip install -U huggingface_hub
```

### 2.2 Download Models
Authenticate and pull the ONNX and FastText binaries:

> Replace <your_access_token> with your HF Token

```bash
hf auth login --token <your_access_token>
```

```bash
hf download Nuvantim/maxim_sentiment_analysis_model maxim-sentiment-models.onnx maxim_fasttext.bin --local-dir models
```
----

## ⚙️ 3. Building the C++ Wrapper
Compile the wrapper code into a shared library to bridge the Go backend with the ONNX runtime.

### 3.1 Extract Dependencies

```bash
7z x lib.7z
```

### 3.2 Compile Shared Object

```bash
g++ -v -O3 -shared -fPIC wrapper/wrapper.cpp -o lib/libwrapper.so -pthread \
    -I./include \
    -L$(pwd)/lib \
    -lonnxruntime \
    -std=c++17 \
    -pthread \
    -Wl,-rpath,'$ORIGIN'
```
----

## 🎯 4. Execution Guide
We use a Makefile to streamline the development and deployment process. Before that, make sure you have copied the application's environment file

```bash
cp env.local .env
``` 

### 🛠 Development Mode
Use this for testing and iterative changes.

> **1. Setup Development Mode**

```bash
make dev
```

> **2. Run the Code**

```bash
make run
```
### 🚀 Production Mode
Use this for final deployment and binary generation.

> **1. Setup Production Mode**

```bash
make prod
```

> **2. Build the Code**

```bash
make build
```

> **3. Run Docker**

```bash
docker compose up -d
```

> **4. Test App**
> 
```bash
curl http://localhost:7860
```
----
