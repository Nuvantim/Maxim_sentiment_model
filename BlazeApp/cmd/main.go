package main

import (
	"api/config"
	"api/internal/server/http"

	"log"
	"os"
	"os/exec"
	"runtime"
)

func ClearScreen() {
	switch runtime.GOOS {
	case "windows":
		cmd := exec.Command("cmd", "/c", "cls")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			log.Fatal(err)
		}
	case "linux", "darwin":
		cmd := exec.Command("/usr/bin/clear")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			log.Fatal(err)
		}
	default:
		log.Fatal("OS not detected")
	}
}

func main() {
	// Check Environment
	if err := config.CheckEnv(); err != nil {
		log.Fatal(err)
	}

	// clear screen
	ClearScreen()

	// Start Server
	app := http.ServerGo()

	// Get Server Config
	serverConfig, err := config.GetServerConfig()
	if err != nil {
		log.Fatal(err)
	}

	done := make(chan struct{}, 1)

	// Start server in goroutine
	go func() {
		if err := app.Listen(":" + serverConfig.Port); err != nil {
			log.Printf("server error: %v", err)
			select {
			case done <- struct{}{}:
			default:
			}
		}
	}()

	go config.GracefulShutdown(app, done)
	<-done
}
