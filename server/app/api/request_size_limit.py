"""ASGI middleware enforcing a hard cap on the request body.

Without this, an oversized upload is fully received and spooled to memory/disk
by Starlette's multipart form parser -- to build the ``UploadFile`` parameter
itself -- before any in-handler check (such as
``image_utils.MAX_FILE_SIZE_BYTES``) or readiness check ever runs. Counting
bytes as they arrive from the ASGI server, before that parsing, closes that gap.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestTooLargeError(Exception):
    """The request body exceeded the configured limit."""


class MaxBodySizeMiddleware:
    """Reject, with 413, any HTTP request whose body exceeds ``max_bytes``.

    Applies to a streamed/chunked body too, since bytes are counted as ASGI
    delivers them rather than read from a ``Content-Length`` header.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        total = 0

        async def guarded_receive() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self._max_bytes:
                    raise RequestTooLargeError(f"request body exceeds {self._max_bytes} bytes")
            return message

        try:
            await self._app(scope, guarded_receive, send)
        except RequestTooLargeError:
            await send(
                {
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b'{"detail":"Request body too large"}'})
