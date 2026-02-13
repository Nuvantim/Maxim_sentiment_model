package handler

import (
	"api/internal/app/request"
	"api/internal/app/service"
	"api/pkg/utils/logs"
	"api/pkg/utils/validates"
	"github.com/gofiber/fiber/v2"
	"github.com/vmihailenco/msgpack/v5"
)

func Home(c *fiber.Ctx) error {
	return c.Render("index", fiber.Map{})
}

func SentimentPrediction(c *fiber.Ctx) error {
	var req request.RequestBody

	if err := msgpack.Unmarshal(c.Body(), &req); err != nil {
		c.Set("Content-Type", "application/x-msgpack")
		return c.Status(400).Send(Response(request.ErrorResponse{err.Error()}))
	}

	words, err := validate.WordFilter(req.Message)
	if err != nil {
		c.Set("Content-Type", "application/x-msgpack")
		return c.Status(400).Send(Response(request.ErrorResponse{err.Error()}))
	}

	req.Message = words

	// Validasi data
	if err := validate.BodyStructs(req); err != nil {
		c.Set("Content-Type", "application/x-msgpack")
		return c.Status(422).Send(Response(request.ErrorResponse{err.Error()}))
	}

	data, err := service.SentimentPrediction(req.Message)
	if err != nil {
		c.Set("Content-Type", "application/x-msgpack")
		return c.Status(500).Send(Response(request.ErrorResponse{err.Error()}))
	}

	go logs.SaveLog(req.Message)

	c.Set("Content-Type", "application/x-msgpack")
	return c.Status(200).Send(Response(data))
}

func Response[T any](data T) []uint8 {
	res, _ := msgpack.Marshal(data)
	return res
}
