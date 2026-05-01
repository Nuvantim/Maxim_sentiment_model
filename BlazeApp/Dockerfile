FROM ubuntu:latest

ENV TERM=xterm-256color

# Set working directory
WORKDIR /app

# Copy Folder
COPY . .

# Give permission
RUN chmod +x bin/api

# Jalankan aplikasi
CMD ["./bin/api"]