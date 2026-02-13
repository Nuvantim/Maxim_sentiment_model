package http

import (
	"api/config"
	"api/internal/routes"

	"github.com/gofiber/fiber/v2"
)

// ServerGo initializes and returns a Fiber app instance
func ServerGo() *fiber.App {
	// Start Fiber APP
	app := fiber.New(config.FiberConfig())
	app.Static("/dist", "./views/dist") //load library frontend

	// Security Configuration
	config.SecurityConfig(app)

	// Set up all routes
	routes.Setup(app)

	// Set Model
	config.LoadFastText()
	config.LoadModel()

	return app
}
