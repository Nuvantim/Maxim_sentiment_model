package config

import (
	"fmt"
	"log"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"

	// "github.com/gofiber/fiber/v2/middleware/csrf"
	"github.com/gofiber/fiber/v2/middleware/helmet"
	"github.com/gofiber/fiber/v2/middleware/idempotency"
	"github.com/gofiber/fiber/v2/middleware/limiter"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/template/html/v2"
)

func FiberConfig() fiber.Config {
	// set folder views as html rendering
	engine := html.New("./views", ".html")

	// Get Environtmet Server Config
	var envServ, err = GetServerConfig()
	if err != nil {
		log.Fatal(err)
	}

	// return fiber config
	return fiber.Config{
		AppName:       envServ.AppName,
		CaseSensitive: true,
		StrictRouting: true,
		ServerHeader:  "Nuvantim Project",
		Prefork:       true,
		Views:         engine,
	}
}

func SecurityConfig(app *fiber.App) {
	// Get Server Config
	serverConfig, err := GetServerConfig()
	if err != nil {
		log.Fatal(err)
	}

	// Rate Limiting
	app.Use(limiter.New(limiter.Config{
		Max:        20,
		Expiration: 120 * time.Second,
	}))

	// Logger
	app.Use(logger.New(logger.Config{
		Format:     "${time} | ${status} | ${latency} | ${ip} | ${method} | ${path}\n",
		TimeFormat: "02-Jan-2006 15:04:05",
		TimeZone:   "Asia/Jakarta",
	}))

	//Idempotency
	app.Use(idempotency.New())

	// CSRF Protection
	// app.Use(csrf.New())

	// helmet
	app.Use(helmet.New(helmet.Config{
		ContentSecurityPolicy: fmt.Sprintf("dafault-src 'self'; frame-ancestors 'none'; http://%s, https://%s", serverConfig.Url, serverConfig.Url),
		HSTSMaxAge:            31536000,
		XFrameOptions:         "DENY",
		HSTSPreloadEnabled:    true,
		HSTSExcludeSubdomains: false,
	}))

	// CORS Configuration
	var origin = fmt.Sprintf("http://%s, https://%s, http://localhost:%s, http://127.0.0.1:%s", serverConfig.Url, serverConfig.Url, serverConfig.Port, serverConfig.Port)
	app.Use(cors.New(cors.Config{
		AllowOrigins:     origin,
		AllowCredentials: true,
		ExposeHeaders:    "Content-Length, X-Knowledge-Base",
		AllowMethods:     "GET,POST,PUT,DELETE",
		AllowHeaders:     "Origin, Content-Type, Authorization, Accept, Accept-Language, Content-Length",
		MaxAge:           3600,
	}))
}
