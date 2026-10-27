"""Bounded byte buffer with backpressure."""

import asyncio
import codecs


class BoundedByteBuffer:
    """Buffer for subprocess output with backpressure.

    Accumulates data up to max_bytes via put(), which awaits when the
    buffer is full. While put() is suspended the caller stops reading
    from the pipe, the kernel buffer fills, and the subprocess blocks
    on write — applying backpressure all the way to the source.
    """

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self._max_bytes = max_bytes
        self._buf = bytearray()
        self._has_space = asyncio.Event()
        self._has_space.set()
        self._closed = False

    async def put(self, data: bytes) -> None:
        """Append data, awaiting if the buffer is full."""
        if not data:
            return
        while len(self._buf) >= self._max_bytes and not self._closed:
            self._has_space.clear()
            await self._has_space.wait()
        if len(self._buf) < self._max_bytes:
            self._buf.extend(data)

    def drain(self, max_bytes: int | None = None) -> bytes:
        """Return buffered data and remove it from the buffer.

        Args:
            max_bytes: Maximum number of bytes to return. If None or
                greater than the buffer size, returns everything.
        """
        if max_bytes is None or max_bytes >= len(self._buf):
            result = bytes(self._buf)
            self._buf.clear()
        else:
            result = bytes(self._buf[:max_bytes])
            del self._buf[:max_bytes]
        self._has_space.set()
        return result

    def close(self) -> None:
        """Unblock any waiting put() so the reader task can exit.

        From then on put() never waits: data is still kept while the buffer
        is under its limit, so a reader can run to EOF with nothing draining
        the buffer, and discarded once it is full.
        """
        self._closed = True
        self._has_space.set()

    @property
    def size(self) -> int:
        """Number of raw bytes available to drain."""
        return len(self._buf)


class DecodingBuffer:
    """Wraps a BoundedByteBuffer with an incremental UTF-8 decoder.

    Ensures multi-byte characters split across drain() calls are
    decoded correctly rather than replaced with U+FFFD.
    """

    def __init__(self, buffer: BoundedByteBuffer) -> None:
        self._buffer = buffer
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def drain(self, final: bool = False, max_bytes: int | None = None) -> str:
        """Drain the underlying buffer and decode to str.

        Args:
            final: If True, flush any trailing partial UTF-8 sequences
                as replacement characters.
            max_bytes: Maximum number of raw bytes to pull from the
                underlying buffer. If None, drains everything.
        """
        return self._decoder.decode(self._buffer.drain(max_bytes), final)
