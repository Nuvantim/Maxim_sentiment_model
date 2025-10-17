# GradioApp
<a href="https://gradio.app">
<img src="https://raw.githubusercontent.com/gradio-app/gradio/207040c4988cdeadbe84e73513902502c1713886/readme_files/gradio.svg">
</a>

Here will explain the instructions for running the model. The model is executed through a web-based interface supported by Gradio.
Please note that we don’t need to run all the models, only a few of them as listed below:

- ```maxim_fasttext.vec```
- ```label_encoder.pkl```
- ```maxim-sentiment-models.onnx```

Make sure you have already [downloaded](https://huggingface.co/Nuvantim/maxim_sentiment_analysis_model/tree/main) it, or you can build the model yourself.
In any case, all models must be stored in the ```models``` folder.

## Python Installation
Here, Python installation is performed on a Linux operating system. If you are using Windows or macOS, you can skip to the next step.

```bash
apt install python3 python3-pip python3-venv
```

## Creating a Python Virtual Environment
```bash
python3 -m venv example && \
source example/bin/activate
```

## Installing the Libraries
```bash
pip install -r requirements.txt
```

## Running the code
```bash
python app.py
```

## Screenshot
<a><img src="https://raw.githubusercontent.com/Nuvantim/Maxim_sentiment_model/refs/heads/main/image/gradio.jpg" border="0"></a>


