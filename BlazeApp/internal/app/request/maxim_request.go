package request

type RequestBody struct {
	Message string `msgpack:"message" validate:"required,min=3,max=250"`
}

type ResponseBody struct {
	Prediction string `msgpack:"prediction"`
}

type ErrorResponse struct {
	Error string `msgpack:"problem"`
}
