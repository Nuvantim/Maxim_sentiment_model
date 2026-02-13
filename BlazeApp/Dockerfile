FROM alpine:latest

# Set working directory
WORKDIR /app

# Copy Folder
COPY ./models ./models
COPY ./lib ./lib
COPY ./bin ./bin
COPY ./views ./views
COPY .env .env

# Memberikan akses eksekusi pada binary
RUN chmod +x /app/bin/api

# Jalankan aplikasi
CMD ["./bin/api"]