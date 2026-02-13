package logs

import (
	"log"
	"os"
)

func SaveLog(word string) {
	f, err := os.OpenFile("log.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("Gagal membuka file log: %v", err)
	} else {
		defer f.Close()
		// Menulis ke log dengan format standar (termasuk tanggal & jam)
		logger := log.New(f, "", log.LstdFlags)
		logger.Println(word)
	}
}
