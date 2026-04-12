import json
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Preprocessing: sessions is a list of lists, where each inner list contains the event_names of a single session

DATA_DIR = "/Users/davitmiroyan/Desktop/CMSC 33231/coauthor-v1.0"
sessions = []
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")]

for fname in files:
    path = os.path.join(DATA_DIR, fname)
    session_events = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                event = json.loads(line)
                session_events.append((event["eventName"], event['eventTimestamp']))
    sessions.append(session_events)

# Unique event types: {'suggestion-reopen', 'suggestion-down', 'suggestion-get', 
# 'suggestion-select', 'text-insert', 'suggestion-up', 'cursor-backward', 
# 'suggestion-hover', 'cursor-forward', 'cursor-select', 'system-initialize', 
# 'text-delete', 'suggestion-open', 'suggestion-close'}


def increasing_text_insert(sessions):
    """
    divides each session into 5 equal time-based chunks using timestamps;
    counts the number of 'text-insert' events in each chunk; determines if
    the frequency of text-insert events decreases over the 5 chunks using
    linear regression; returns the proportion of sessions that have a
    decreasing text-insert frequency along with the statistical significance
    of that result
    """
    N_CHUNKS = 5
    increasing_sessions = []

    for session in sessions:

        start_time = session[0][1]
        end_time = session[-1][1]
        duration = end_time - start_time

        chunk_duration = duration / N_CHUNKS
        chunk_counts = [0] * N_CHUNKS # stores the number of text-insert events per chunk

        for event_name, timestamp in session:
            chunk_index = min(int((timestamp - start_time) / chunk_duration), N_CHUNKS - 1) # chunk_index determines the chunk in which the event should be stored
            if event_name == "text-insert":
                chunk_counts[chunk_index] += 1

        x = np.arange(N_CHUNKS)
        slope, _ = np.polyfit(x, chunk_counts, 1) # linear regression on the 5 counts

        if slope > 0:
            increasing_sessions.append(session)

    n_total = len(sessions)
    n_increasing = len(increasing_sessions)

    result = stats.binomtest(n_increasing, n_total, p=0.5, alternative='greater') # test for statistical significance

    print(f"Total sessions: {n_total}")
    print(f"Increasing text-insert frequency: {n_increasing} ({100*n_increasing/n_total:.1f}%)")
    print(f"p-value: {result.pvalue:.4f}")


def decreasing_suggestion_get(sessions):
    """
    divides each session into 5 equal time-based chunks using timestamps;
    counts the number of 'suggestion-get' events in each chunk; determines if
    the frequency of suggestion-get events decreases over the 5 chunks using
    linear regression; returns the proportion of sessions that have a
    decreasing suggestion-get frequency along with the statistical significance
    of that result
    """
    N_CHUNKS = 5
    decreasing_sessions = []

    for session in sessions:

        start_time = session[0][1]
        end_time = session[-1][1]
        duration = end_time - start_time

        chunk_duration = duration / N_CHUNKS
        chunk_counts = [0] * N_CHUNKS # stores the number of suggestion-get events per chunk

        for event_name, timestamp in session:
            chunk_index = min(int((timestamp - start_time) / chunk_duration), N_CHUNKS - 1)
            if event_name == "suggestion-get":
                chunk_counts[chunk_index] += 1

        x = np.arange(N_CHUNKS)
        slope, _ = np.polyfit(x, chunk_counts, 1) # linear regression on the 5 counts

        if slope < 0:
            decreasing_sessions.append(session)

    n_total = len(sessions)
    n_decreasing = len(decreasing_sessions)

    result = stats.binomtest(n_decreasing, n_total, p=0.5, alternative='greater') # test for statistical significance

    print(f"Total sessions: {n_total}")
    print(f"Decreasing suggestion-get frequency: {n_decreasing} ({100*n_decreasing/n_total:.1f}%)")
    print(f"p-value: {result.pvalue:.4f}")


def correlation_suggestion_get_text_insert(sessions):
    """
    divides each session into 5 equal time-based chunks using timestamps;
    computes the slope of 'suggestion-get' frequency and 'text-insert' frequency
    over the 5 chunks for each session using linear regression; computes the
    Spearman correlation between the two slopes across all sessions; a negative
    correlation would indicate that as users write more themselves over time,
    they seek suggestions less
    """
    N_CHUNKS = 5
    suggestion_slopes = []
    text_insert_slopes = []

    for session in sessions:

        start_time = session[0][1]
        end_time = session[-1][1]
        duration = end_time - start_time

        chunk_duration = duration / N_CHUNKS
        suggestion_counts = [0] * N_CHUNKS # stores the number of suggestion-get events per chunk
        text_insert_counts = [0] * N_CHUNKS # stores the number of text-insert events per chunk

        for event_name, timestamp in session:
            chunk_index = min(int((timestamp - start_time) / chunk_duration), N_CHUNKS - 1)
            if event_name == "suggestion-get":
                suggestion_counts[chunk_index] += 1
            elif event_name == "text-insert":
                text_insert_counts[chunk_index] += 1

        x = np.arange(N_CHUNKS)
        suggestion_slope, _ = np.polyfit(x, suggestion_counts, 1) # linear regression on suggestion-get counts
        text_insert_slope, _ = np.polyfit(x, text_insert_counts, 1) # linear regression on text-insert counts

        suggestion_slopes.append(suggestion_slope)
        text_insert_slopes.append(text_insert_slope)

    correlation, p_value = stats.spearmanr(suggestion_slopes, text_insert_slopes) # spearman correlation between the two slopes

    print(f"Total sessions: {len(suggestion_slopes)}")
    print(f"Spearman correlation: {correlation:.4f}")
    print(f"p-value: {p_value:.4f}")


def suggestion_window_engagement(sessions):
    """
    for each suggestion window (suggestion-open to suggestion-close), counts
    the number of events in between; filters by mode: 'all', 'accepted', or 'rejected';
    fits a linear regression to those counts across the session to determine if
    the user spends less time engaging with suggestion windows over time;
    returns the proportion of sessions with a decreasing trend along with the
    statistical significance of that result
    """
    assert mode in ("all", "accepted", "rejected"), "mode must be 'all', 'accepted', or 'rejected'"
    
    decreasing_sessions = []
    valid_sessions = []

    for session in sessions:
        window_event_counts = [] # stores the number of events between each open-close pair

        i = 0
        while i < len(session):
            if session[i][0] == "suggestion-open":
                j = i + 1
                events_in_between = 0
                while j < len(session):
                    if session[j][0] == "suggestion-select":
                        events_in_between += 1
                        j += 1
                    elif session[j][0] == "suggestion-close":
                        break
                    else:
                        events_in_between += 1
                        j += 1
                if j < len(session):
                        window_event_counts.append(events_in_between)
                i = j + 1
            else:
                i += 1

        if len(window_event_counts) < 2:
            continue

        valid_sessions.append(session)

        x = np.arange(len(window_event_counts))
        slope, _ = np.polyfit(x, window_event_counts, 1) # linear regression on the counts

        if slope < 0:
            decreasing_sessions.append(session)

    n_total = len(valid_sessions)
    n_decreasing = len(decreasing_sessions)

    result = stats.binomtest(n_decreasing, n_total, p=0.5, alternative='greater') # test for statistical significance

    print(f"Total sessions: {n_total}")
    print(f"Decreasing window engagement: {n_decreasing} ({100*n_decreasing/n_total:.1f}%)")
    print(f"p-value: {result.pvalue:.4f}")


def plot_engagement_session(sessions):

    N_CHUNKS = 5
    all_text_insert = []
    all_suggestion_get = []
    
    for session in sessions:
        start_time = session[0][1]
        end_time = session[-1][1]
        duration = end_time - start_time
        chunk_duration = duration / N_CHUNKS
    
        ti_counts = [0] * N_CHUNKS
        sg_counts = [0] * N_CHUNKS
    
        for event_name, timestamp in session:
            chunk_index = min(int((timestamp - start_time) / chunk_duration), N_CHUNKS - 1)
            if event_name == "text-insert":
                ti_counts[chunk_index] += 1
            elif event_name == "suggestion-get":
                sg_counts[chunk_index] += 1
    
        all_text_insert.append(ti_counts)
        all_suggestion_get.append(sg_counts)
    
    avg_ti = np.mean(all_text_insert, axis=0)
    avg_sg = np.mean(all_suggestion_get, axis=0)
    
    x = np.arange(1, N_CHUNKS + 1)
    
    fig, ax1 = plt.subplots(figsize=(7, 4))
    
    color_ti = "#2196F3"
    color_sg = "#F44336"
    
    ax1.plot(x, avg_ti, color=color_ti, marker='o', linewidth=2, label="text-insert")
    ax1.set_xlabel("Session chunk", fontsize=12)
    ax1.set_ylabel("Avg. text-insert count", color=color_ti, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color_ti)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    ax2 = ax1.twinx()
    ax2.plot(x, avg_sg, color=color_sg, marker='s', linewidth=2, linestyle='--', label="suggestion-get")
    ax2.set_ylabel("Avg. suggestion-get count", color=color_sg, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color_sg)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    
    plt.title("text-insert vs. suggestion-get frequency over a writing session", fontsize=12)
    fig.tight_layout()
    plt.savefig("/Users/davitmiroyan/Desktop/plot_exp1_exp2.png", dpi=150)
    plt.close()
    print("Saved plot_exp1_exp2.png")
 
 
def plot_engagement_window(sessions):
 
    all_window_counts = []
    
    for session in sessions:
        window_event_counts = []
        i = 0
        while i < len(session):
            if session[i][0] == "suggestion-open":
                j = i + 1
                events_in_between = 0
                while j < len(session):
                    if session[j][0] == "suggestion-close":
                        break
                    else:
                        events_in_between += 1
                        j += 1
                if j < len(session):
                    window_event_counts.append(events_in_between)
                i = j + 1
            else:
                i += 1
    
        if len(window_event_counts) >= 2:
            all_window_counts.append(window_event_counts)
    
    # Normalize to relative position (0–1) and bin into 10 bins
    N_BINS = 10
    bin_sums = [0.0] * N_BINS
    bin_counts = [0] * N_BINS
    
    for window_counts in all_window_counts:
        n = len(window_counts)
        for idx, count in enumerate(window_counts):
            relative_pos = idx / (n - 1) if n > 1 else 0
            bin_index = min(int(relative_pos * N_BINS), N_BINS - 1)
            bin_sums[bin_index] += count
            bin_counts[bin_index] += 1
    
    avg_window = [bin_sums[i] / bin_counts[i] if bin_counts[i] > 0 else 0 for i in range(N_BINS)]
    x_bins = np.arange(1, N_BINS + 1)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_bins, avg_window, color="#4CAF50", marker='o', linewidth=2)
    ax.set_xlabel("Relative position in session", fontsize=12)
    ax.set_ylabel("Avg. events per suggestion window", fontsize=11)
    ax.set_title("Suggestion window engagement over a writing session", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.tight_layout()
    plt.savefig("/Users/davitmiroyan/Desktop/plot_exp3.png", dpi=150)
    plt.close()
    print("Saved plot_exp3.png")
