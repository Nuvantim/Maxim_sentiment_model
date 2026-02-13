package config

import (
	"errors"

	"github.com/joho/godotenv"
	"github.com/kelseyhightower/envconfig"
)

type ServerConfig struct {
	Port    string `envconfig:"PORT"`
	AppName string `envconfig:"APP_NAME"`
	Url     string `envconfig:"URL"`
}

func CheckEnv() error {
	if err := godotenv.Load(); err != nil {
		return errors.New(err.Error())
	}
	return nil
}

func GetServerConfig() (*ServerConfig, error) {
	var serv ServerConfig
	if err := envconfig.Process("", &serv); err != nil {
		return nil, err
	}
	return &serv, nil
}
