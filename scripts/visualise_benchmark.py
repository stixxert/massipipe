#!/usr/bin/env python3
"""Plot total or mean timings from MassiPipe profile-event JSONL files.

Examples:
    python scripts/visualise_benchmark.py baseline.jsonl optimized.jsonl \
        --labels baseline optimized --output benchmark-comparison.png
    python scripts/visualise_benchmark.py baseline-1.jsonl baseline-2.jsonl \
        optimized-1.jsonl optimized-2.jsonl \
        --labels baseline baseline optimized optimized --statistic total
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from massipipe.timing import aggregate_tag_files

NANOSECONDS_PER_SECOND = 1_000_000_000

# Spans recorded one level under a phase, used to split a phase bar into parts.
SPAN_TAGS = ("data_load", "data_compute", "data_save")
SPAN_COLORS = {
    "data_load": "#4C78A8",
    "data_compute": "#F58518",
    "data_save": "#54A24B",
}
SPAN_LABELS = {"data_load": "load", "data_compute": "compute", "data_save": "save"}
REMAINDER_COLOR = "#BAB0AC"
PROFILE_HATCHES = ("", "///", "...", "xxx")
# Ignore leftover time below this fraction of a bar's total (floating-point dust).
SPAN_TOLERANCE = 1e-9
# A child row carrying at least this share of its parent repeats the parent's bar.
WRAPPER_SHARE = 0.99


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare total or mean wall and process CPU timings from one or more "
            "MassiPipe profile-event JSONL files."
        )
    )
    parser.add_argument(
        "profiles",
        nargs="+",
        type=Path,
        metavar="EVENTS.jsonl",
        help="profile-event JSONL files produced inside timing.profile_events()",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="labels for inputs; repeat a label to combine multiple runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-profile.png"),
        help="chart output path (default: benchmark-profile.png)",
    )
    parser.add_argument(
        "--tag-prefix",
        help="slash-separated tag prefix to include, e.g. pipeline/mosaic",
    )
    parser.add_argument(
        "--statistic",
        choices=("total", "mean"),
        default="total",
        help=(
            "plot total time across all root runs or mean time for roots containing each tag "
            "(default: total)"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="show the top N tag paths by wall time; use 0 to show all (default: 25)",
    )
    args = parser.parse_args(argv)

    if args.labels is not None and len(args.labels) != len(args.profiles):
        parser.error("--labels must provide one label for each profile file")
    if args.top < 0:
        parser.error("--top must be zero or a positive integer")
    if args.tag_prefix is not None and any(not part for part in args.tag_prefix.split("/")):
        parser.error("--tag-prefix must contain non-empty slash-separated tag components")

    resolved_profiles = {path.resolve() for path in args.profiles}
    if args.output.resolve() in resolved_profiles:
        parser.error("--output must not overwrite an input profile")

    for profile_path in args.profiles:
        if not profile_path.is_file():
            parser.error(f"profile file does not exist or is not a file: {profile_path}")

    return args


def load_profiles(
    paths: list[Path],
    labels: list[str] | None,
    tag_prefix: tuple[str, ...] | None,
) -> list[tuple[str, dict[tuple[str, ...], dict[str, Any]]]]:
    grouped_paths: dict[str, list[Path]] = {}
    if labels is None:
        for index, path in enumerate(paths):
            label = path.stem
            if label in grouped_paths:
                label = f"{path.parent.name}/{path.stem}"
            suffix = index + 1
            while label in grouped_paths:
                label = f"{path} ({suffix})"
                suffix += 1
            grouped_paths[label] = [path]
    else:
        for path, label in zip(paths, labels, strict=True):
            grouped_paths.setdefault(label, []).append(path)

    profiles = []
    for label, grouped in grouped_paths.items():
        summaries = aggregate_tag_files(
            [str(path) for path in grouped],
            tag_prefix=tag_prefix,
        )
        by_tag = {tuple(summary["tags"]): summary for summary in summaries}
        if not by_tag:
            raise ValueError(f"no profile events matched for {label}")
        profiles.append((label, by_tag))
    return profiles


def _lighten(color: Any, amount: float) -> tuple[float, float, float]:
    """Blend a colour towards white (amount 0.0 keeps it, 1.0 makes it white)."""
    red, green, blue = to_rgb(color)
    return tuple(channel + (1.0 - channel) * amount for channel in (red, green, blue))


def _tag_children(
    summaries: dict[tuple[str, ...], dict[str, Any]],
) -> dict[tuple[str, ...], list[tuple[str, ...]]]:
    """Index each recorded tag path by its parent, so a phase can be walked down."""

    children: dict[tuple[str, ...], list[tuple[str, ...]]] = {
        tags: [] for tags in summaries
    }
    for tags in summaries:
        if tags and tags[:-1] in children:
            children[tags[:-1]].append(tags)
    return children


def _category_segments(
    summaries: dict[tuple[str, ...], dict[str, Any]],
    children: dict[tuple[str, ...], list[tuple[str, ...]]],
    tags: tuple[str, ...],
    metric: str,
    statistic: str,
) -> list[tuple[str | None, float]]:
    """Split one tag path's time into load / compute / save spans plus a remainder.

    Each recorded tag path owns its exclusive (child-free) time, and those own
    times telescope to the total of the tag path being drawn, so the segments
    always add up to the bar they are stacked in. A span takes the colour of its
    last tag component; everything else - a phase's own work, or an uninstrumented
    child stage - is the neutral remainder.
    """

    totals: dict[str | None, float] = {}

    def accumulate(tag_path: tuple[str, ...]) -> None:
        total = _duration_seconds(summaries.get(tag_path), metric, statistic)[0]
        if not math.isfinite(total) or total <= 0.0:
            return

        child_paths = children.get(tag_path, ())
        child_total = 0.0
        for child in child_paths:
            child_time = _duration_seconds(summaries.get(child), metric, statistic)[0]
            if math.isfinite(child_time):
                child_total += child_time
            accumulate(child)

        own_time = total - child_total
        # Subtract child time with a relative tolerance so floating-point dust from
        # fully-explained tag paths does not become a visible remainder segment.
        if own_time > total * SPAN_TOLERANCE:
            category = tag_path[-1] if tag_path and tag_path[-1] in SPAN_TAGS else None
            totals[category] = totals.get(category, 0.0) + own_time

    accumulate(tags)
    return [(span, totals[span]) for span in (*SPAN_TAGS, None) if totals.get(span, 0.0) > 0.0]


def _duration_seconds(
    summary: dict[str, Any] | None,
    metric: str,
    statistic: str,
) -> tuple[float, float, float]:
    if summary is None:
        return math.nan, 0.0, 0.0

    if statistic == "total":
        return float(summary[f"{metric}_total"]) / NANOSECONDS_PER_SECOND, 0.0, 0.0

    mean = float(summary[f"{metric}_mean"]) / NANOSECONDS_PER_SECOND
    minimum = float(summary[f"{metric}_min"]) / NANOSECONDS_PER_SECOND
    maximum = float(summary[f"{metric}_max"]) / NANOSECONDS_PER_SECOND
    return mean, max(0.0, mean - minimum), max(0.0, maximum - mean)


def _stage_rows(tags: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Drop load / compute / save rows that are already part of a parent stage bar.

    Those spans are drawn as coloured segments inside the stage they belong to, so a
    separate row would just repeat the same time at a smaller scale.
    """

    available = set(tags)
    return [
        tag_path
        for tag_path in tags
        if not (tag_path and tag_path[-1] in SPAN_TAGS and tag_path[:-1] in available)
    ]


def _drop_wrapper_rows(
    tags: list[tuple[str, ...]],
    profiles: list[tuple[str, dict[tuple[str, ...], dict[str, Any]]]],
    metric: str,
) -> list[tuple[str, ...]]:
    """Hide a child row that carries (almost) all of its parent's time.

    A stage that only wraps a single instrumented call records the same duration
    twice; the parent row already shows it, so the wrapper row adds nothing.
    """

    available = {tag_path for _, summaries in profiles for tag_path in summaries}
    kept = []
    for tag_path in tags:
        parent = tag_path[:-1]
        if parent in available and all(
            _share_of_parent(summaries, tag_path, parent, metric) >= WRAPPER_SHARE
            for _, summaries in profiles
        ):
            continue
        kept.append(tag_path)
    return kept


def _share_of_parent(
    summaries: dict[tuple[str, ...], dict[str, Any]],
    tag_path: tuple[str, ...],
    parent: tuple[str, ...],
    metric: str,
) -> float:
    parent_time = float(summaries.get(parent, {}).get(metric, 0.0))
    if parent_time <= 0.0:
        return 0.0
    return float(summaries.get(tag_path, {}).get(metric, 0.0)) / parent_time


def plot_profiles(
    profiles: list[tuple[str, dict[tuple[str, ...], dict[str, Any]]]],
    output_path: Path,
    top: int,
    statistic: str,
) -> list[tuple[str, ...]]:
    all_tags = set().union(*(set(summaries) for _, summaries in profiles))
    ranking_metric = f"wall_time_ns_{statistic}"
    ranked_tags = sorted(
        all_tags,
        key=lambda tags: (
            -max(summaries.get(tags, {}).get(ranking_metric, 0) for _, summaries in profiles),
            tags,
        ),
    )
    shown_tags = _stage_rows(ranked_tags)
    shown_tags = _drop_wrapper_rows(shown_tags, profiles, ranking_metric)
    shown_tags = shown_tags[:top] if top else shown_tags

    if not shown_tags:
        raise ValueError("no tag paths to plot")

    figure_height = max(4.5, 0.42 * len(shown_tags) + 2.0)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    y_positions = list(range(len(shown_tags)))
    group_height = 0.8
    metric_series = (
        ("wall_time_ns", "wall", 0.0),
        ("cpu_time_ns", "process CPU", 0.5),
    )
    series_count = len(profiles) * len(metric_series)
    bar_height = group_height / series_count
    time_label = "Total" if statistic == "total" else "Mean"

    # One bar per phase: split into its load / compute / save spans and shaded
    # solid for wall time, light for process CPU time, so both the composition of
    # a phase and its waiting-on-I/O can be read inside the same bar.
    series_index = 0
    for profile_index, (label, summaries) in enumerate(profiles):
        children = _tag_children(summaries)
        hatch = PROFILE_HATCHES[profile_index % len(PROFILE_HATCHES)]
        for metric, _, lighten in metric_series:
            offset = (series_index - (series_count - 1) / 2) * bar_height
            for position, tags in zip(y_positions, shown_tags):
                left = 0.0
                segments = _category_segments(summaries, children, tags, metric, statistic)
                for span, value in segments:
                    color = SPAN_COLORS.get(span, REMAINDER_COLOR)
                    axis.barh(
                        position + offset,
                        value,
                        left=left,
                        height=bar_height * 0.88,
                        color=_lighten(color, lighten),
                        edgecolor=color,
                        linewidth=0.5,
                        hatch=hatch,
                    )
                    left += value

                if statistic == "mean":
                    mean, lower, upper = _duration_seconds(summaries.get(tags), metric, statistic)
                    if math.isfinite(mean):
                        axis.errorbar(
                            mean,
                            position + offset,
                            xerr=[[lower], [upper]],
                            fmt="none",
                            ecolor="#404040",
                            elinewidth=0.8,
                            capsize=2,
                        )
            series_index += 1

    axis.set_title("wall time (solid) vs process CPU time (light), split by load / compute / save")
    axis.set_xlabel(f"{time_label} time (s)")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0)

    axis.set_yticks(y_positions)
    axis.set_yticklabels([" / ".join(tags) for tags in shown_tags])
    axis.invert_yaxis()

    figure.suptitle(f"MassiPipe benchmark profile — {time_label.lower()} time", y=0.995)
    span_handles = [
        Patch(facecolor=SPAN_COLORS[span], edgecolor=SPAN_COLORS[span], label=SPAN_LABELS[span])
        for span in SPAN_TAGS
    ]
    span_handles.append(Patch(facecolor=REMAINDER_COLOR, edgecolor=REMAINDER_COLOR, label="rest"))

    # Figure legends sit above the axes, outside tight_layout's reach, so their
    # bands are reserved in pixels and converted to the figure's fractions; that
    # keeps the gap constant no matter how many tag paths are drawn.
    height_px = figure.get_figheight() * figure.dpi
    legend_y = 1.0 - 20.0 / height_px
    figure.legend(
        handles=span_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=len(span_handles),
    )
    layout_top = 1.0 - 17.0 / height_px

    if len(profiles) > 1:
        # Colour already encodes the span, so profiles are told apart by hatching.
        profile_handles = [
            Patch(
                facecolor="white",
                edgecolor="#404040",
                hatch=PROFILE_HATCHES[index % len(PROFILE_HATCHES)],
                label=label,
            )
            for index, (label, _) in enumerate(profiles)
        ]
        figure.legend(
            handles=profile_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y - 32.0 / height_px),
            ncol=min(len(profile_handles), 4),
        )
        layout_top = 1.0 - 49.0 / height_px

    figure.tight_layout(rect=(0, 0, 1, layout_top))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return shown_tags


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tag_prefix = tuple(args.tag_prefix.split("/")) if args.tag_prefix else None

    try:
        profiles = load_profiles(args.profiles, args.labels, tag_prefix)
        shown_tags = plot_profiles(profiles, args.output, args.top, args.statistic)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Wrote {args.output} ({len(shown_tags)} tag paths, {len(profiles)} profile(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
