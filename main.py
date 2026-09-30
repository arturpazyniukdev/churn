import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.routes import router
from core import state
from schemas import ErrorResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("churn")

state.load()

app = FastAPI()
app.include_router(router)


def error_response(status: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(code=code, message=message, details=details).model_dump(),
    )


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    log.warning("http error %d: %s", exc.status_code, exc.detail)
    return error_response(exc.status_code, "http_error", exc.detail)


@app.exception_handler(RequestValidationError)
def validation_handler(request: Request, exc: RequestValidationError):
    log.warning("validation error: %s", exc.errors())
    return error_response(422, "validation_error", "invalid request data", details=exc.errors())


@app.exception_handler(Exception)
def unhandled_handler(request: Request, exc: Exception):
    log.exception("unhandled error")
    return error_response(500, "internal_error", "unexpected server error", details=str(exc))
