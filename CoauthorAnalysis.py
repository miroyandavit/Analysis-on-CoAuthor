import json
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


DATA_DIR = "/Users/davitmiroyan/Desktop/CMSC 33231/coauthor-v1.0"


def load_sessions(data_dir):
    """
    Loads all sessions from the CoAuthor dataset. Each session is a list of
    (event_name, timestamp) tuples, sorted by file order.
    """
    sessions = []
    files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    for fname in files:
        path = os.path.join(data_dir, fname)
        session_events = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    event = json.loads(line)
                    session_events.append((event["eventName"], event["eventTimestamp"]))
        sessions.append(session_events)
    return sessions


def chunk_session(session, n_chunks=5):
    """
    Divides a session into n equal time-based chunks.
    Returns a list of chunk indices for each event.
    """
    start_time = session[0][1]
    end_time = session[-1][1]
    duration = end_time - start_time
    chunk_duration = duration / n_chunks

    indices = []
    for _, timestamp in session:
        chunk_index = min(int((timestamp - start_time) / chunk_duration), n_chunks - 1)
        indices.append(chunk_index)
    return indices


def increasing_text_insert(sessions, n_chunks=5):
    """
    For each session, counts 'text-insert' events per time chunk and fits a
    linear regression over the chunks. A positive slope indicates that the
    writer types more independently as the session progresses.
    Reports the proportion of sessions with an increasing trend and its
    statistical significance via a one-sided binomial test.
    """
    increasing_sessions = 0

    for session in sessions:
        chunk_indices = chunk_session(session, n_chunks)
        chunk_counts = [0] * n_chunks

        for (event_name, _), chunk_index in zip(session, chunk_indices):
            if event_name == "text-insert":
                chunk_counts[chunk_index] += 1

        slope, _ = np.polyfit(np.arange(n_chunks), chunk_counts, 1)
        if slope > 0:
            increasing_sessions += 1

    n_total = len(sessions)
    result = stats.binomtest(increasing_sessions, n_total, p=0.5, alternative="greater")

    print(f"Total sessions: {n_total}")
    print(f"Increasing text-insert frequency: {increasing_sessions} ({100 * increasing_sessions / n_total:.1f}%)")
    print(f"p-value: {result.pvalue:.4f}")


def decreasing_suggestion_get(sessions, n_chunks=5):
    """
    For each session, counts 'suggestion-get' events per time chunk and fits a
    linear regression over the chunks. A negative slope indicates that the
    writer requests AI suggestions less frequently as the session progresses.
    Reports the proportion of sessions with a decreasing trend and its
    statistical significance via a one-sided binomial test.
    """
    decreasing_sessions = 0

    for session in sessions:
        chunk_indices = chunk_session(session, n_chunks)
        chunk_counts = [0] * n_chunks

        for (event_name, _), chunk_index in zip(session, chunk_indices):
            if event_name == "suggestion-get":
                chunk_counts[chunk_index] += 1

        slope, _ = np.polyfit(np.arange(n_chunks), chunk_counts, 1)
        if slope < 0:
            decreasing_sessions += 1

    n_total = len(sessions)
    result = stats.binomtest(decreasing_sessions, n_total, p=0.5, alternative="greater")

    print(f"Total sessions: {n_total}")
    print(f"Decreasing suggestion-get frequency: {decreasing_sessions} ({100 * decreasing_sessions / n_total:.1f}%)")
    print(f"p-value: {result.pvalue:.4f}")


def correlation_suggestion_get_text_insert(sessions, n_chunks=5):
    """
    For each session, computes the linear regression slope of both
    'suggestion-get' and 'text-insert' frequencies over time chunks.
    Computes the Spearman correlation between the two slopes across all
    sessions. A negative correlation indicates that sessions where AI usage
    decreases are the same sessions where independent writing increases.
    """
    suggestion_slopes = []
    text_insert_slopes = []

    for session in sessions:
        chunk_indices = chunk_session(session, n_chunks)
        suggestion_counts = [0] * n_chunks
        text_insert_counts = [0] * n_chunks

        for (event_name, _), chunk_index in zip(session, chunk_indices):
            if event_name == "suggestion-get":
                suggestion_counts[chunk_index] += 1
            elif event_name == "text-insert":
                text_insert_counts[chunk_index] += 1

        x = np.arange(n_chunks)
        suggestion_slopes.append(np.polyfit(x, suggestion_counts, 1)[0])
        text_insert_slopes.append(np.polyfit(x, text_insert_counts, 1)[0])

    correlation, p_value = stats.spearmanr(suggestion_slopes, text_insert_slopes)

    print(f"Total sessions: {len(sessions)}")
    print(f"Spearman correlation: {correlation:.4f}")
    print(f"p-value: {p_value:.4f}")


def suggestion_window_engagement(sessions):
    """
    For each suggestion window (suggestion-open to suggestion-close), measures
    the time elapsed as a proxy for how deeply the writer engages with the
    suggestion. Fits a linear regression over windows within each session.
    A negative slope indicates the writer spends less time considering
    suggestions as the session progresses. Reports the proportion of sessions
    with a decreasing trend and its statistical significance via a one-sided
    binomial test. Sessions with fewer than 2 suggestion windows are excluded.
    """
    decreasing_sessions = 0
    valid_sessions = 0

    for session in sessions:
        window_durations = []

        i = 0
        while i < len(session):
            if session[i][0] == "suggestion-open":
                open_time = session[i][1]
                j = i + 1
                while j < len(session):
                    if session[j][0] == "suggestion-close":
                        window_durations.append(session[j][1] - open_time)
                        break
                    j += 1
                i = j + 1
            else:
                i += 1

        if len(window_durations) < 2:
            continue

        valid_sessions += 1
        slope, _ = np.polyfit(np.arange(len(window_durations)), window_durations, 1)
        if slope < 0:
            decreasing_sessions += 1

    result = stats.binomtest(decreasing_sessions, valid_sessions, p=0.5, alternative="greater")

    print(f"Total sessions: {valid_sessions}")
    print(f"Decreasing window engagement: {decreasing_sessions} ({100 * decreasing_sessions / valid_sessions:.1f}%)")
    print(f"p-value: {result.pvalue:.4f}")


def plot_engagement_over_session(sessions, output_path, n_chunks=5):
    """
    Plots the average 'text-insert' and 'suggestion-get' counts per time chunk
    across all sessions on a dual-axis chart, illustrating the inverse
    relationship between independent writing and AI usage over a session.
    """
    all_text_insert = []
    all_suggestion_get = []

    for session in sessions:
        chunk_indices = chunk_session(session, n_chunks)
        ti_counts = [0] * n_chunks
        sg_counts = [0] * n_chunks

        for (event_name, _), chunk_index in zip(session, chunk_indices):
            if event_name == "text-insert":
                ti_counts[chunk_index] += 1
            elif event_name == "suggestion-get":
                sg_counts[chunk_index] += 1

        all_text_insert.append(ti_counts)
        all_suggestion_get.append(sg_counts)

    avg_ti = np.mean(all_text_insert, axis=0)
    avg_sg = np.mean(all_suggestion_get, axis=0)
    x = np.arange(1, n_chunks + 1)

    color_ti = "#2196F3"
    color_sg = "#F44336"

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, avg_ti, color=color_ti, marker="o", linewidth=2, label="text-insert")
    ax1.set_xlabel("Session chunk", fontsize=12)
    ax1.set_ylabel("Avg. text-insert count", color=color_ti, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_ti)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))

    ax2 = ax1.twinx()
    ax2.plot(x, avg_sg, color=color_sg, marker="s", linewidth=2, linestyle="--", label="suggestion-get")
    ax2.set_ylabel("Avg. suggestion-get count", color=color_sg, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_sg)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)

    plt.title("text-insert vs. suggestion-get frequency over a writing session", fontsize=12)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    sessions = load_sessions(DATA_DIR)

    print("=== Experiment 1: Decreasing suggestion-get frequency ===")
    decreasing_suggestion_get(sessions)

    print("\n=== Experiment 2: Increasing text-insert frequency ===")
    increasing_text_insert(sessions)

    print("\n=== Experiment 2: Correlation between suggestion-get and text-insert slopes ===")
    correlation_suggestion_get_text_insert(sessions)

    print("\n=== Experiment 3: Decreasing suggestion window engagement ===")
    suggestion_window_engagement(sessions)

    plot_engagement_over_session(sessions, output_path="/Users/davitmiroyan/Desktop/plot_exp1_exp2.png")
