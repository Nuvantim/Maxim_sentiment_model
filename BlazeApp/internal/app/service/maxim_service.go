package service

import (
	"api/internal/app/request"
	"api/wrapper"
	"fmt"
)

func SentimentPrediction(word string) (request.ResponseBody, error) {
	label, akurasi, err := wrapper.ModelPrediction(word)
	if err != nil {
		return request.ResponseBody{}, err
	}

	var resp request.ResponseBody

	switch label {
	case 1:
		resp.Prediction = fmt.Sprintf("POSITIF(%.2f%%)", akurasi)
		return resp, nil
	case 0:
		resp.Prediction = fmt.Sprintf("NEGATIF(%.2f%%)", akurasi)
		return resp, nil
	default:
		return request.ResponseBody{}, fmt.Errorf("failed prediction: %s", word)
	}

}
