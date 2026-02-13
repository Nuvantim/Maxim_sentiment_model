package config

import (
	"context"
	"github.com/gofiber/fiber/v2"

	"log"
	"os/signal"
	"syscall"
	"time"
)

func GracefulShutdown(app *fiber.App, done chan struct{}) {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	<-ctx.Done()

	log.Println("Shutting down gracefully...")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := app.ShutdownWithContext(shutdownCtx); err != nil {
		log.Printf("Server forced to shutdown with error: %v", err)
	}

	log.Println("Server exiting")

	done <- struct{}{}
}
