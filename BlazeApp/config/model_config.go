package config

import (
	"fmt"
	"log"
	"os"
)

func LoadModel() {
	if _, err := os.Stat("models/maxim-sentiment-models.onnx"); err != nil {
		log.Fatalf("[⛔] Failed load GRU model : %s", err.Error())
	}
	fmt.Println("[🔣] FastText model is available...✅")
}

func LoadFastText() {
	if _, err := os.Stat("models/maxim_fasttext.bin"); err != nil {
		log.Fatalf("[⛔] Failed load FastText model : %s", err.Error())
	}
	fmt.Println("[🧠] GRU model is available...✅")
}
