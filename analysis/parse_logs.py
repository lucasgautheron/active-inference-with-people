#!/usr/bin/env python3
"""
Simple Log Parser to DataFrame

This script parses log files to extract timing information and creates a pandas DataFrame.

Usage: python log_parser.py <log_file_path>
"""

import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("pgf")
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
    },
)
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def parse_log_file(file_path):
    """Parse the log file and return a pandas DataFrame."""

    # Regex patterns
    participant_pattern = re.compile(r"Preparing trial for participant (\d+)")
    prior_data_pattern = re.compile(r"Task 'prior_data' took ([\d.]+) s")
    prioritize_nodes_pattern = re.compile(
        r"Task 'prioritize_nodes' took ([\d.]+) s"
    )
    loop_condition_pattern = re.compile(
        r"while_loop \((optimal_(?:treatment|test))__trial_loop\)"
    )
    timestamp_pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)"
    )

    data = []
    current_participant = None
    current_timestamp = None
    current_entry = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            # Extract timestamp
            timestamp_match = timestamp_pattern.match(line)
            if timestamp_match:
                current_timestamp = timestamp_match.group(1)

            # Check for new participant
            participant_match = participant_pattern.search(line)
            if participant_match:
                # Save previous entry if it has data
                if current_participant is not None and current_entry:
                    data.append(
                        {
                            "timestamp": current_timestamp,
                            "participant_id": current_participant,
                            "prior_data_time": current_entry.get("prior_data"),
                            "prioritize_nodes_time": current_entry.get(
                                "prioritize_nodes"
                            ),
                            "task_nature": current_entry.get("task_nature"),
                        }
                    )

                # Start new entry
                current_participant = int(participant_match.group(1))
                current_entry = {}
                continue

            # Skip if no current participant
            if current_participant is None:
                continue

            # Extract timings and task nature
            prior_data_match = prior_data_pattern.search(line)
            if prior_data_match:
                current_entry["prior_data"] = float(prior_data_match.group(1))

            prioritize_match = prioritize_nodes_pattern.search(line)
            if prioritize_match:
                current_entry["prioritize_nodes"] = float(
                    prioritize_match.group(1)
                )

            loop_match = loop_condition_pattern.search(line)
            if loop_match:
                current_entry["task_nature"] = loop_match.group(1)

        # Don't forget the last entry
        if current_participant is not None and current_entry:
            data.append(
                {
                    "timestamp": current_timestamp,
                    "participant_id": current_participant,
                    "prior_data_time": current_entry.get("prior_data"),
                    "prioritize_nodes_time": current_entry.get(
                        "prioritize_nodes"
                    ),
                    "task_nature": current_entry.get("task_nature"),
                }
            )

    return pd.DataFrame(data)


def plot_performance(df, task):
    fig, ax = plt.subplots(figsize=(3.2, 2.13333))
    optimal_test = df[df["task_nature"] == f"optimal_{task}"]
    optimal_test.reset_index(inplace=True)
    optimal_test["no"] = optimal_test.index
    optimal_test.dropna(inplace=True)
    x = optimal_test["no"]
    ax.fill_between(
        x,
        np.zeros(len(x)),
        optimal_test["prior_data_time"],
        alpha=0.5,
        label="I/O",
    )
    ax.fill_between(
        x,
        optimal_test["prior_data_time"],
        optimal_test["prioritize_nodes_time"],
        alpha=0.5,
        label="Computations",
    )

    low = np.quantile(optimal_test["prioritize_nodes_time"], q=0.05 / 2)
    high = np.quantile(optimal_test["prioritize_nodes_time"], q=1 - 0.05 / 2)
    ax.text(
        0.05,
        0.95,
        f"$\\tau \in [{low:.1f} s,{high:.1f} s]_{{95\\%}}$",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Time in seconds")
    ax.set_ylim(0, 2.5)

    fig.legend(ncol=2, loc="upper center", frameon=False)
    fig.savefig(f"output/performance_{task}.pdf", bbox_inches="tight")


def main():
    df = parse_log_file("output/lucas-adaptive-run1.log")
    plot_performance(df, "test")
    plot_performance(df, "treatment")


if __name__ == "__main__":
    main()
