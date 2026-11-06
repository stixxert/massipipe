from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterable, Iterator, Mapping, TextIO, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
_active_log: _EventLog | None = None
_active_lock = threading.Lock()
_tag_context = threading.local()


@dataclass
class _DurationTotals:
    count: int = 0
    total: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0

    def add(self, duration_ns: float) -> None:
        self.count += 1
        self.total += duration_ns
        self.minimum = duration_ns if self.count == 1 else min(duration_ns, self.minimum)
        self.maximum = duration_ns if self.count == 1 else max(duration_ns, self.maximum)

    def fields(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_total": self.total,
            f"{prefix}_min": self.minimum,
            f"{prefix}_max": self.maximum,
            f"{prefix}_mean": self.total / self.count,
        }


class _EventLog:
    def __init__(self, output: TextIO) -> None:
        self.output = output
        self.lock = threading.Lock()
        self.next_id = 1

    def flush(self) -> None:
        with self.lock:
            self.output.flush()

    @contextmanager
    def call(
        self,
        tags: tuple[str, ...],
        function_id: str,
        root_call_id: int | None,
    ) -> Iterator[int]:
        with self.lock:
            call_id = self.next_id
            self.next_id += 1

        root_call_id = call_id if root_call_id is None else root_call_id

        # perf_counter_ns is monotonic and intended for measuring elapsed time.
        start_ns = time.perf_counter_ns()
        # process_time_ns measures CPU consumed by this process, excluding sleep.
        start_cpu_ns = time.process_time_ns()
        try:
            yield call_id
        finally:
            end_ns = time.perf_counter_ns()
            end_cpu_ns = time.process_time_ns()

            event = {
                "call_id": call_id,
                "function_id": function_id,
                "root_call_id": root_call_id,
                "tags": tags,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "start_cpu_ns": start_cpu_ns,
                "end_cpu_ns": end_cpu_ns,
                "thread_id": threading.get_ident(),
            }

            line = json.dumps(event, separators=(",", ":"))

            with self.lock:
                self.output.write(line + "\n")


@contextmanager
def profile_events(output: TextIO) -> Iterator[None]:
    """Track decorated call events while the profile latch is active."""

    global _active_log

    with _active_lock:
        if _active_log is not None:
            raise RuntimeError("Nested profile event session")

        log = _EventLog(output)
        _active_log = log

    try:
        yield
    finally:
        with _active_lock:
            _active_log = None
        log.flush()


def profiled(tag: str) -> Callable[[F], F]:
    """Tag a function level. Nesting decorated profiled calls add parent tags."""

    if not isinstance(tag, str) or not tag:
        raise ValueError("tag must be a non-empty string")

    def decorator(function: F) -> F:
        function_id = f"{function.__qualname__}"

        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with _active_lock:
                log = _active_log
            if log is None:
                return function(*args, **kwargs)

            parent = getattr(_tag_context, "path", ())
            parent_root_call_id = getattr(_tag_context, "root_call_id", None)

            full_path = parent + (tag,)
            _tag_context.path = full_path

            try:
                with log.call(full_path, function_id, parent_root_call_id) as call_id:
                    _tag_context.root_call_id = (
                        call_id if parent_root_call_id is None else parent_root_call_id
                    )
                    return function(*args, **kwargs)
            finally:
                _tag_context.path = parent

                if parent_root_call_id is None:
                    try:
                        del _tag_context.root_call_id
                    except AttributeError:
                        pass
                else:
                    _tag_context.root_call_id = parent_root_call_id

        return wrapped

    return decorator


def validate_event_identity(
    event: Mapping[str, Any], *, location: str = "event"
) -> tuple[str, int]:
    """Validate function and root IDs required for aggregation."""

    missing = [field for field in ("function_id", "root_call_id") if field not in event]

    if missing:
        fields = " and ".join(missing)
        raise ValueError(f"{location}: missing required field(s): {fields}")

    function_id = event["function_id"]
    root_call_id = event["root_call_id"]

    if not isinstance(function_id, str) or not function_id:
        raise ValueError(f"{location}: function_id must be a non-empty string")

    if type(root_call_id) is not int:
        raise ValueError(f"{location}: root_call_id must be an integer")

    return function_id, root_call_id


def reduce_tag_samples(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return total wall/CPU time per root invocation and exact tag path."""

    by_tag: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}

    for event in events:
        _, root_call_id = validate_event_identity(event)
        key = (root_call_id, tuple(event["tags"]))
        sample = by_tag.setdefault(
            key,
            {"call_ids": [], "wall_ns": 0, "cpu_ns": 0},
        )

        wall_ns = event["end_ns"] - event["start_ns"]
        cpu_ns = event["end_cpu_ns"] - event["start_cpu_ns"]
        if wall_ns < 0 or cpu_ns < 0:
            raise ValueError("event timestamps must be monotonically increasing")

        sample["call_ids"].append(event["call_id"])
        sample["wall_ns"] += wall_ns
        sample["cpu_ns"] += cpu_ns

    return [
        {
            "root_call_id": root_call_id,
            "tags": tags,
            **sample,
        }
        for (root_call_id, tags), sample in sorted(by_tag.items())
    ]


def aggregate_tags(
    path: str,
    *,
    tag_prefix: Iterable[str] | str | None = None,
    exact_tag_path: Iterable[str] | str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate by full path, with optional filters for tag prefixes or exact paths."""

    if tag_prefix is not None and exact_tag_path is not None:
        raise ValueError("tag_prefix and exact_tag_path are mutually exclusive")

    selected = tag_prefix if tag_prefix is not None else exact_tag_path

    if isinstance(selected, str):
        raise ValueError("tag filters must be a sequence of tag components")

    selected_path = tuple(selected) if selected is not None else ()

    if selected is not None and not selected_path:
        raise ValueError("tag filters must not be empty")

    exact = exact_tag_path is not None

    events = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            events.append(json.loads(line))

    groups: dict[tuple[str, ...], tuple[_DurationTotals, _DurationTotals]] = {}

    for sample in reduce_tag_samples(events):
        tags = sample["tags"]

        if selected_path:
            matches = (
                tags == selected_path if exact else tags[: len(selected_path)] == selected_path
            )
            if not matches:
                continue

        wall_ns = sample["wall_ns"]
        cpu_ns = sample["cpu_ns"]

        if wall_ns < 0 or cpu_ns < 0:
            raise ValueError("event timestamps not monotonically increasing")

        totals = groups.get(tags, None)

        if totals is None:
            totals = (_DurationTotals(), _DurationTotals())

            groups[tags] = totals

        totals[0].add(wall_ns)
        totals[1].add(cpu_ns)

    results = []

    for tags, (wall, cpu) in sorted(groups.items()):
        result = {"tags": list(tags), "call_count": wall.count}

        result.update(wall.fields("wall_time_ns"))
        result.update(cpu.fields("cpu_time_ns"))

        results.append(result)

    return results


def aggregate_tag_files(
    paths: Iterable[str],
    *,
    tag_prefix: Iterable[str] | None = None,
    exact_tag_path: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Combine tag summaries from multiple profile files without mixing run IDs."""

    if tag_prefix is not None and exact_tag_path is not None:
        raise ValueError("tag_prefix and exact_tag_path are mutually exclusive")
    if isinstance(tag_prefix, str) or isinstance(exact_tag_path, str):
        raise ValueError("tag filters must be a sequence of tag components")

    prefix = tuple(tag_prefix) if tag_prefix is not None else None
    exact = tuple(exact_tag_path) if exact_tag_path is not None else None
    if prefix == () or exact == ():
        raise ValueError("tag filters must not be empty")

    by_tag: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for path in paths:
        for summary in aggregate_tags(
            path,
            tag_prefix=prefix,
            exact_tag_path=exact,
        ):
            by_tag.setdefault(tuple(summary["tags"]), []).append(summary)

    results = []
    for tags, summaries in sorted(by_tag.items()):
        result: dict[str, Any] = {"tags": list(tags)}
        result["call_count"] = sum(summary["call_count"] for summary in summaries)

        for metric in ("wall_time_ns", "cpu_time_ns"):
            total = sum(summary[f"{metric}_total"] for summary in summaries)
            result[f"{metric}_total"] = total
            result[f"{metric}_min"] = min(summary[f"{metric}_min"] for summary in summaries)
            result[f"{metric}_max"] = max(summary[f"{metric}_max"] for summary in summaries)
            result[f"{metric}_mean"] = total / result["call_count"]

        results.append(result)

    return results
