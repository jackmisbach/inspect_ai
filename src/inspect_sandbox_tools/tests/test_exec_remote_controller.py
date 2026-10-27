"""Unit tests for the exec_remote Controller and Job.

Controller tests inject mock Jobs. Job tests spawn short-lived real
subprocesses to exercise kill and shutdown behaviour.
"""

import asyncio
import os
import shlex
import signal
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from inspect_sandbox_tools._remote_tools._exec_remote import _job as job_module
from inspect_sandbox_tools._remote_tools._exec_remote._controller import Controller
from inspect_sandbox_tools._remote_tools._exec_remote._job import Job
from inspect_sandbox_tools._remote_tools._exec_remote.tool_types import (
    KillResult,
    PollResult,
)
from inspect_sandbox_tools._util.common_types import ToolException


class TestControllerConcurrentPollAndKill:
    """Verify that concurrent poll() and kill() on the same PID don't raise."""

    @pytest.mark.asyncio
    async def test_concurrent_poll_and_kill_does_not_raise(self) -> None:
        """
        Concurrent poll (returning completed) and kill for the same PID should both succeed without raising KeyError.

        The race: both coroutines call _get_job(pid) successfully, then both
        await their respective job methods (yielding control), then both try
        to delete the job from _jobs. With bare `del`, the second one raises
        KeyError.
        """
        controller = Controller()
        pid = 42

        completed_result = PollResult(
            state="completed", exit_code=0, seq=1, stdout="", stderr=""
        )

        # Create a mock job whose poll() and kill() yield control via
        # asyncio.sleep(0). This ensures both coroutines get past _get_job()
        # before either attempts deletion.
        job = MagicMock()
        job.pid = pid
        job.cleanup = AsyncMock()

        async def mock_poll(ack_seq: int) -> PollResult:
            await asyncio.sleep(0)
            return completed_result

        async def mock_kill(ack_seq: int) -> tuple[int, str, str]:
            await asyncio.sleep(0)
            return (1, "", "")

        job.poll = mock_poll
        job.kill = mock_kill

        # Inject the mock job directly into the controller's registry.
        controller._jobs[pid] = job

        # Both poll and kill will try to del self._jobs[pid].
        # Run them concurrently — only one should do the deletion.
        poll_result, kill_result = await asyncio.gather(
            controller.poll(pid, ack_seq=0),
            controller.kill(pid, ack_seq=0),
        )

        # Both should complete without error.
        assert poll_result.state == "completed"
        assert kill_result.stdout == ""

        # The job should have been removed from the registry.
        assert pid not in controller._jobs

        # cleanup should have been called exactly once, not twice.
        assert job.cleanup.call_count == 1


@pytest.mark.asyncio
async def test_poll_of_unknown_pid_names_the_missing_job() -> None:
    """The host's exec_remote client matches this message after a timed-out poll."""
    controller = Controller()

    with pytest.raises(ToolException, match=r"^No job found with pid 42"):
        await controller.poll(42, ack_seq=0)


@pytest.mark.asyncio
async def test_shutdown_terminates_and_removes_all_jobs() -> None:
    controller = Controller()
    jobs = []
    for pid in (41, 42):
        job = MagicMock()
        job.shutdown = AsyncMock()
        job.cleanup = AsyncMock()
        controller._jobs[pid] = job
        jobs.append(job)

    await controller.shutdown()

    assert controller._jobs == {}
    for job in jobs:
        job.shutdown.assert_awaited_once_with()
        job.cleanup.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_shutdown_reports_failures_from_every_job() -> None:
    controller = Controller()
    for pid, message in ((41, "first shutdown failed"), (42, "second shutdown failed")):
        job = MagicMock()
        job.shutdown = AsyncMock(side_effect=RuntimeError(message))
        job.cleanup = AsyncMock()
        controller._jobs[pid] = job

    with pytest.raises(RuntimeError) as error:
        await controller.shutdown()

    assert "first shutdown failed" in str(error.value)
    assert "second shutdown failed" in str(error.value)


@pytest.mark.asyncio
async def test_retired_job_shutdown_uses_only_its_captured_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = await asyncio.create_subprocess_exec("true")
    await process.wait()
    job = Job(process)
    captured_child = MagicMock(pid=99)
    capture_group = MagicMock(return_value=[captured_child])
    terminate = AsyncMock()
    monkeypatch.setattr(job_module, "process_group_members", capture_group)
    monkeypatch.setattr(job_module, "terminate_process_tree", terminate)

    job.retire()
    await job.shutdown()

    assert process.pid is not None
    capture_group.assert_called_once_with(process.pid, exclude_pid=process.pid)
    terminate.assert_awaited_once_with(
        process,
        timeout=30,
        process_group=False,
        known_descendants=[captured_child],
    )


async def _poll_until_done(
    controller: Controller, pid: int, ack_seq: int = 0
) -> tuple[PollResult, str]:
    """Poll like the host does, returning the terminal reply and all stdout."""
    stdout = ""
    while True:
        result = await controller.poll(pid, ack_seq)
        assert len(result.stdout) <= job_module._MAX_POLL_OUTPUT_BYTES
        stdout, ack_seq = stdout + result.stdout, result.seq
        if result.state != "running":
            return result, stdout
        await anyio.sleep(0.05)  # a host round trip between polls


def _wait_for_file(path: Path) -> str:
    """A shell loop that ends once the test creates `path`."""
    return f"while [ ! -e {shlex.quote(str(path))} ]; do sleep 0.01; done"


async def _wait_for_leader_exit(job: Job) -> None:
    """Wait for the leader to exit without polling, so no poll has seen the exit."""
    while job._process.returncode is None:
        await anyio.sleep(0.01)


async def test_poll_keeps_output_a_child_writes_after_the_leader_exits() -> None:
    """Buffers close only after the readers reach EOF, not when the leader exits."""
    controller = Controller()
    pid = await controller.submit("(sleep 0.3; printf late) & printf early")
    try:
        with anyio.fail_after(5):
            result, stdout = await _poll_until_done(controller, pid)
    finally:
        await controller.shutdown()

    assert result.state == "completed"
    assert stdout == "earlylate"


async def test_poll_completes_while_a_child_floods_the_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that keeps writing after the leader exits cannot delay completion."""
    monkeypatch.setattr(job_module, "_READER_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(job_module, "_BACKPRESSURE_BUFFER_SIZE", 4 * 1024 * 1024)
    controller = Controller()
    pid = await controller.submit("yes & sleep 0.3; exit 3")  # the pipe is full
    try:
        with anyio.fail_after(10):
            result, _ = await _poll_until_done(controller, pid)
    finally:
        await controller.shutdown()

    assert result.state == "completed"
    assert result.exit_code == 3


async def test_poll_waits_one_grace_period_for_both_streams() -> None:
    """The readers share one grace period rather than each waiting for its own."""
    controller = Controller()
    pid = await controller.submit("sleep 30 & exit 0")  # the child holds both pipes
    job = controller._jobs[pid]
    try:
        with anyio.fail_after(10):
            await _wait_for_leader_exit(job)
            start = anyio.current_time()
            result, stdout = await _poll_until_done(controller, pid)
            elapsed = anyio.current_time() - start
    finally:
        await controller.shutdown()

    assert result.state == "completed"
    assert stdout == ""
    grace = job_module._READER_GRACE_SECONDS
    assert grace - 0.05 <= elapsed < 1.75 * grace  # one stream after the other: 2x


async def test_poll_sends_a_full_reply_at_once_and_readers_keep_their_grace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A full reply buffered at exit is sent without waiting; the readers go on."""
    monkeypatch.setattr(job_module, "_READER_GRACE_SECONDS", 30.0)
    full = job_module._MAX_POLL_OUTPUT_BYTES
    size = 3 * full
    release = tmp_path / "release"
    controller = Controller()
    pid = await controller.submit(
        f"head -c {size} /dev/zero; ({_wait_for_file(release)}; printf late) & exit 0"
    )
    job = controller._jobs[pid]
    try:
        with anyio.fail_after(10):  # well inside the grace period
            await _wait_for_leader_exit(job)
            while job._buffered() < full:
                await anyio.sleep(0.01)
            first = await controller.poll(pid, 0)
        assert first.state == "running"
        assert len(first.stdout) == full
        assert not job._stdout_task.done()  # the readers were not stopped

        release.touch()
        with anyio.fail_after(10):
            result, rest = await _poll_until_done(controller, pid, first.seq)
    finally:
        release.touch()
        await controller.shutdown()

    assert result.state == "completed"
    assert first.stdout + rest == "\0" * size + "late"


async def test_poll_cancelled_while_readers_settle_keeps_the_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A poll cancelled while waiting for the readers leaves the job to the next poll."""
    monkeypatch.setattr(job_module, "_READER_GRACE_SECONDS", 60.0)
    settling = anyio.Event()
    settle_readers = Job._settle_readers

    async def signal_then_settle(job: Job) -> None:
        settling.set()
        await settle_readers(job)

    monkeypatch.setattr(Job, "_settle_readers", signal_then_settle)
    release = tmp_path / "release"
    # the child holds the pipes, and writes, only once the test creates `release`
    controller = Controller()
    pid = await controller.submit(
        f"({_wait_for_file(release)}; printf late) & printf early"
    )
    job = controller._jobs[pid]

    async def poll_until_settling() -> None:
        while True:  # each poll before the leader exits returns at once
            await controller.poll(pid, 0)
            await anyio.sleep(0.01)

    try:
        with anyio.fail_after(5):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(poll_until_settling)
                await settling.wait()
                tasks.cancel_scope.cancel()

        # the controller still owns the job, and its readers were not stopped
        assert controller._jobs[pid] is job
        assert not job._stdout_task.done()
        release.touch()
        with anyio.fail_after(5):
            result, stdout = await _poll_until_done(controller, pid)
    finally:
        release.touch()
        await controller.shutdown()

    assert result.state == "completed"
    assert stdout == "earlylate"
    assert pid not in controller._jobs
    assert job._stdout_task.done() and job._stderr_task.done()


async def _stop_job(job: Job) -> None:
    if job._process.returncode is None:
        job._process.kill()
        await job._process.wait()
    await job.cleanup()


@pytest.mark.asyncio
async def test_kill_signals_group_while_leader_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = await Job.create("sleep 30")
    signalled: list[tuple[int, int]] = []
    real_killpg = os.killpg

    def record_killpg(pgid: int, sig: int) -> None:
        signalled.append((pgid, sig))
        real_killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", record_killpg)
    try:
        await job.kill(ack_seq=0)
        assert job._process.returncode is not None
    finally:
        await _stop_job(job)

    assert signalled == [(job.pid, signal.SIGTERM)]


@pytest.mark.asyncio
async def test_kill_after_leader_exited_returns_output_without_signalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = await Job.create("echo done")
    await job._process.wait()
    # Process exit does not guarantee the readers have buffered its output.
    await job._stdout_task
    await job._stderr_task
    killpg = MagicMock(side_effect=AssertionError("signalled a stale process group"))
    monkeypatch.setattr(os, "killpg", killpg)

    try:
        seq, stdout, stderr = await job.kill(ack_seq=0)
    finally:
        await _stop_job(job)

    killpg.assert_not_called()
    assert seq == 1
    assert stdout == "done\n"
    assert stderr == ""


async def test_kill_keeps_output_a_child_writes_after_the_leader_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """kill() after the leader exited, but before a poll saw it, waits for the child."""
    waiting = anyio.Event()
    wait_for_readers = Job._wait_for_readers

    async def signal_then_wait(job: Job) -> None:
        waiting.set()
        await wait_for_readers(job)

    monkeypatch.setattr(Job, "_wait_for_readers", signal_then_wait)
    release = tmp_path / "release"
    controller = Controller()
    pid = await controller.submit(
        f"({_wait_for_file(release)}; printf late) & printf early"
    )
    job = controller._jobs[pid]
    killed: list[KillResult] = []

    async def kill() -> None:
        killed.append(await controller.kill(pid, 0))

    try:
        with anyio.fail_after(5):
            await _wait_for_leader_exit(job)
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(kill)
                await waiting.wait()  # kill() is waiting for the readers
                release.touch()
    finally:
        release.touch()
        await controller.shutdown()

    assert [result.stdout for result in killed] == ["earlylate"]


@pytest.mark.asyncio
async def test_kill_treats_reused_pid_as_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leader handle whose identity no longer matches means the PID was reused."""
    job = await Job.create("sleep 30")
    job._leader = MagicMock(is_running=MagicMock(return_value=False))
    killpg = MagicMock()
    monkeypatch.setattr(os, "killpg", killpg)

    try:
        await job.kill(ack_seq=0)
    finally:
        await _stop_job(job)

    killpg.assert_not_called()


@pytest.mark.asyncio
async def test_kill_does_not_escalate_to_a_dead_leaders_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The leader dies on SIGTERM while a descendant in its own session holds the pipes.

    On Python 3.11+ ``Process.wait()`` then stays blocked past the grace period,
    and the SIGKILL escalation must not signal the dead leader's group id.
    """
    pidfile = tmp_path / "grandchild.pid"
    grandchild = (
        "import os,pathlib,time; os.setsid(); "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(300)"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(300)"
    )
    job = await Job.create(f"{shlex.quote(sys.executable)} -c {shlex.quote(parent)}")
    for _ in range(200):
        if pidfile.exists():
            break
        await asyncio.sleep(0.05)
    assert pidfile.exists(), "grandchild did not start"

    signalled: list[int] = []
    real_killpg = os.killpg

    def record_killpg(pgid: int, sig: int) -> None:
        signalled.append(sig)
        real_killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", record_killpg)
    try:
        await asyncio.wait_for(job.kill(ack_seq=0, timeout=1), 10)
    finally:
        try:
            os.kill(int(pidfile.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass
        await _stop_job(job)

    assert signalled == [signal.SIGTERM]
    assert job._process.returncode is not None
