"""Request-owned commands with bounded waits."""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from time import monotonic
from typing import NamedTuple

from inspect_sandbox_tools._util.common_types import ToolException

from ._job import Job
from .tool_types import KillResult, PollResult, StartWaitParams, WaitResult

_EXPIRED_ID_TTL = 3600.0
_MAX_EXPIRED_IDS = 50_000  # about 13 MB
_LAUNCH_SHUTDOWN_GRACE = 1.0


@dataclass
class _Request:
    job: Job | None = None
    launch: asyncio.Task[None] | None = None
    stop: asyncio.Task[KillResult] | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    # Serialize output/ACK mutations; never hold this while waiting for new output.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancelled: bool = False
    error: Exception | None = None
    response: PollResult | None = None
    finished_at: float | None = None


class _Expired(NamedTuple):
    at: float
    outcome: int | str
    """The exit code of a command that completed, or what else happened to it."""


def _outcome(entry: _Request) -> int | str:
    response = entry.response
    if response is not None and response.exit_code is not None:
        return response.exit_code
    if response is not None and response.state == "killed":
        return "ran and was killed"
    if entry.error is not None and entry.job is None:
        return f"could not be started ({entry.error!r})"
    if entry.error is not None:
        return f"was stopped when its input could not be written ({entry.error!r})"
    return "was cancelled before it started"


class WaitController:
    """Own launches independently of RPC cancellation.

    Request IDs must never be reused by clients. A repeated start with a known
    ID reports that request instead of launching again. Retention is described
    in `_prune`; a missing entry is an explicit error, never an implicit new launch.
    A restart forgets every ID, so a start repeated after one launches again.
    """

    def __init__(self) -> None:
        self._requests: dict[str, _Request] = {}
        self._expired: OrderedDict[str, _Expired] = OrderedDict()
        self._retired_jobs: list[Job] = []

    async def start(
        self, params: StartWaitParams, can_switch_user: bool = False
    ) -> WaitResult:
        """Register a unique request before launching and wait briefly for completion."""
        self._prune()
        if params.request_id in self._requests or params.request_id in self._expired:
            # A retried start (e.g. after a dropped connection) acts as a poll; a
            # cancelled or expired request raises instead of launching late.
            return await self.poll(params.request_id, 0, params.wait_seconds)
        if params.stdin_open:
            raise ToolException("Awaitable exec_remote requests cannot keep stdin open")
        entry = _Request()
        self._requests[params.request_id] = entry
        entry.launch = asyncio.create_task(self._launch(entry, params, can_switch_user))
        await asyncio.shield(entry.launch)
        job = self._job(entry, params.request_id)
        await job.wait_for_update(params.wait_seconds)
        return await self._poll(entry, params.request_id, 0)

    async def _launch(
        self, entry: _Request, params: StartWaitParams, can_switch_user: bool
    ) -> None:
        def register(job: Job) -> None:
            entry.job = job
            entry.ready.set()

        try:
            await Job.create(
                params.command,
                input=params.input,
                stdin_open=params.stdin_open,
                env=params.env,
                cwd=params.cwd,
                user=params.user,
                can_switch_user=can_switch_user,
                on_create=register,
            )
        except Exception as ex:
            if not entry.cancelled:
                entry.error = ex
                entry.finished_at = monotonic()
        finally:
            entry.ready.set()

    async def poll(
        self, request_id: str, ack_seq: int, wait_seconds: float
    ) -> WaitResult:
        """Wait for output/exit, replaying an unacknowledged response immediately."""
        entry = self._get(request_id)
        if entry.launch is not None:
            await entry.ready.wait()
        job = self._job(entry, request_id)
        if entry.response is None or ack_seq >= entry.response.seq:
            await job.wait_for_update(wait_seconds)
        return await self._poll(entry, request_id, ack_seq)

    async def _poll(self, entry: _Request, request_id: str, ack_seq: int) -> WaitResult:
        async with entry.lock:
            job = self._job(entry, request_id)
            if entry.response is not None and ack_seq < entry.response.seq:
                result = entry.response
            else:
                result = await job.poll(ack_seq, terminal_output=True)
                entry.response = result
            if result.state != "running" and entry.finished_at is None:
                entry.finished_at = monotonic()
                job.retire()
                await job.cleanup()
            response = WaitResult(pid=job.pid, result=result)
            self._prune()
            return response

    async def kill(self, request_id: str, ack_seq: int) -> KillResult:
        """Cancel a launch even when the start RPC has not registered it yet."""
        self._prune()
        entry = self._requests.get(request_id)
        if entry is None:
            if request_id not in self._expired:
                self._requests[request_id] = _Request(
                    cancelled=True, finished_at=monotonic()
                )
                self._prune()
            return KillResult(seq=0, stdout="", stderr="")
        entry.cancelled = True
        if entry.stop is None:
            entry.stop = asyncio.create_task(self._stop(entry, ack_seq))
            entry.stop.add_done_callback(lambda task: self._stop_finished(entry, task))
        result = await asyncio.shield(entry.stop)
        # A repeated kill with a newer ACK must not return output already delivered.
        if ack_seq >= result.seq:
            return KillResult(seq=result.seq, stdout="", stderr="")
        return result

    @staticmethod
    def _stop_finished(entry: _Request, task: asyncio.Task[KillResult]) -> None:
        # A cancelled RPC must not retain an unaccounted copy of its output.
        if entry.stop is task:
            entry.stop = None
        if not task.cancelled():
            task.exception()

    async def _stop(self, entry: _Request, ack_seq: int) -> KillResult:
        if entry.launch is not None:
            await entry.ready.wait()
        async with entry.lock:
            if entry.job is None:
                if entry.finished_at is None:
                    entry.finished_at = monotonic()
                self._prune()
                return KillResult(seq=0, stdout="", stderr="")
            seq, stdout, stderr = await entry.job.kill(ack_seq)
            entry.response = PollResult(
                state="killed", seq=seq, stdout=stdout, stderr=stderr
            )
            if entry.finished_at is None:
                entry.finished_at = monotonic()
            entry.job.retire()
            await entry.job.cleanup()
            result = KillResult(seq=seq, stdout=stdout, stderr=stderr)
            self._prune()
            return result

    def _get(self, request_id: str) -> _Request:
        self._prune()
        entry = self._requests.get(request_id)
        if entry is None:
            if (expired := self._expired.get(request_id)) is not None:
                outcome = expired.outcome
                if isinstance(outcome, int):
                    outcome = f"ran and exited with code {outcome}"
                raise ToolException(
                    f"exec_remote request {request_id} result lost: the command "
                    f"{outcome}. Its output is no longer kept, and the command is "
                    "not run again."
                )
            raise ToolException(f"Unknown or expired exec_remote request {request_id}")
        return entry

    @staticmethod
    def _job(entry: _Request, request_id: str) -> Job:
        if entry.error is not None:
            # Raised as exec_remote_start raises it, so the host sees the same error.
            raise entry.error
        if entry.job is None:
            raise ToolException(
                f"exec_remote request {request_id} was cancelled before it started"
            )
        return entry.job

    def _prune(self) -> None:
        """Forget finished requests, keeping their IDs.

        A finished request's terminal reply is not kept once it has been sent:
        its entry is dropped. Its ID is kept, with how the command ended but
        without output, for `_EXPIRED_ID_TTL` (at most `_MAX_EXPIRED_IDS` of
        them), so a late repeated start or poll fails, saying the result was
        lost and how the command ended, instead of launching it again.
        """
        for key, entry in list(self._requests.items()):
            if entry.finished_at is not None:
                self._evict(key, entry)
        now = monotonic()
        while self._expired and (
            len(self._expired) > _MAX_EXPIRED_IDS
            or now - next(iter(self._expired.values())).at >= _EXPIRED_ID_TTL
        ):
            self._expired.popitem(last=False)

    def _evict(self, key: str, entry: _Request) -> None:
        self._requests.pop(key, None)
        self._expired[key] = _Expired(monotonic(), _outcome(entry))
        if entry.job is not None:
            entry.job.retire()
            entry.job.release_output()
            self._retired_jobs.append(entry.job)

    async def shutdown(self) -> None:
        """Terminate pending, active, and retired commands owned by this controller."""
        entries = list(self._requests.values())
        for entry in entries:
            entry.cancelled = True
        # Starting cancellation before awaiting launch also unblocks stdin.drain().
        results = await asyncio.gather(
            *(self._stop(entry, 0) for entry in entries), return_exceptions=True
        )
        jobs = set(entry.job for entry in entries if entry.job is not None) | set(
            self._retired_jobs
        )
        # Before awaiting launches: a launch can be blocked writing input to a
        # descendant that holds stdin open after the leader exited.
        shutdown_results = await asyncio.gather(
            *(job.shutdown() for job in jobs), return_exceptions=True
        )
        launches = [entry.launch for entry in entries if entry.launch is not None]
        if launches:
            # A process that left the job's process group can still hold stdin.
            _, blocked = await asyncio.wait(launches, timeout=_LAUNCH_SHUTDOWN_GRACE)
            for launch in blocked:
                launch.cancel()
            await asyncio.gather(*launches, return_exceptions=True)
        self._requests.clear()
        self._retired_jobs.clear()
        errors = [
            result
            for result in [*results, *shutdown_results]
            if isinstance(result, Exception)
        ]
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))
