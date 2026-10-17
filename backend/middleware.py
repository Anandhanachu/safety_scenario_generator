"""Bound JSON request bodies before framework parsing (including chunked requests)."""
from starlette.responses import JSONResponse


class BodyLimit:
    def __init__(self, app, limit=1024 * 1024):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.limit:
                return await JSONResponse({"detail": "Request body too large"}, status_code=413)(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        async def buffered():
            if chunks:
                return chunks.pop(0)
            return await receive()
        await self.app(scope, buffered, send)
