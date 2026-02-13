package routes

import (
	"api/internal/app/handler"
	"github.com/gofiber/fiber/v2"
)

func Setup(app *fiber.App) {
	app.Get("/", handler.Home)
	app.Post("/sentiment-prediction", handler.SentimentPrediction)

}
