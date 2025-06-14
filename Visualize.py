import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import pandas as pd
import os


"""
Functions for visualizing results of the APTCID test.  Needs to be modified for extensibility.

FOR METRIC-FREQUENCY PLOT
bins = np.linspace(0, 1, 101)
probs_cols =['RBC_model_prob', 'FFP_model_prob', 'PLT_model_prob']
lab_cols = ['outcome_prbc', 'outcome_ffp', 'outcome_platelet']
class_freq_list = compute_class_frequencies_per_column(data_valid,probs_cols,lab_cols, x)

"""

def compute_freq_list_from_df(df, proba_cols, x):
    bins = x
    freq_list = []

    for col in proba_cols:
        values = df[col].dropna()
        freq, _ = np.histogram(values, bins=bins)
        # Pad with one extra value (e.g., zero) to match x.shape
        freq = np.append(freq, 0)  # OR use np.insert(freq, 0, 0) if you want to prepend
        freq_list.append(freq)

    return freq_list


#-----------------------


def compute_class_frequencies_per_column(df, prob_cols, label_cols, thresholds):
    """
    Compute binned frequency counts for class 0 and 1 to match shape of `thresholds`.

    Parameters:
    - df: DataFrame with probabilities and labels
    - prob_cols: list of probability column names
    - label_cols: list of binary label column names
    - thresholds: 1D array of thresholds (must be length N)

    Returns:
    - class_freq_list: list of (class0_counts, class1_counts) arrays, each of shape (N,)
    """
    bins = thresholds  # assume thresholds already has shape (N,)
    class_freq_list = []

    for prob_col, label_col in zip(prob_cols, label_cols):
        sub_df = df[[prob_col, label_col]].dropna()

        class0 = sub_df[sub_df[label_col] == 0]
        class1 = sub_df[sub_df[label_col] == 1]

        class0_counts, _ = np.histogram(class0[prob_col], bins=bins)
        class1_counts, _ = np.histogram(class1[prob_col], bins=bins)

        # Pad both with one extra 0 to match len(thresholds)
        class0_counts = np.append(class0_counts, 0)
        class1_counts = np.append(class1_counts, 0)

        class_freq_list.append((class0_counts, class1_counts))

    return class_freq_list


#------------------------Metrics and Class Frequencies---------------------#
"""
Plot designed to adhere to Oxford University Press Figure guidelines. 
Saves to png/pdf.
"""

def plot_oup_line_and_freq_subplots(
    x,
    y_list,
    ci_list,
    labels,
    titles,
    freq_list,
    n_cols=3,
    figsize=(16, 6),
    bar_color="gray",
    vlines=None,
    save_path_base=None
):
    """
    Create a 2-row, N-column figure:
    - Row 1: Line plots with confidence intervals
    - Row 2: Frequency bar plots

    Parameters:
    - x: 1D array of x-axis values
    - y_list: list of lists of y-arrays (shape: [n_cols][3])
    - ci_list: list of 3-tuples (lower, upper) per metric per column
    - labels: list of 3-label lists per column
    - titles: subplot titles for top row
    - freq_list: list of arrays (one per column) for frequency bar plots
    - n_cols: number of subplot columns (default = 3)
    - figsize: overall figure size
    - bar_color: barplot color for bottom row
    - save_path_base: if given, saves both PDF and PNG at this base path
    - vlines: list of lists of tuples [(x1, label1), (x2, label2), ...] per subplot
    """
    set_oup_figure_style()
    fig, axs = plt.subplots(2, n_cols, figsize=figsize, sharex='col', sharey='row')

    # Colors and styles
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    linestyles = ['solid', 'dashed', 'dotted']

    for i in range(n_cols):
        ax_top = axs[0, i]
        ax_bottom = axs[1, i]

        vlines_i = vlines[i] if vlines is not None else []

        # Line plot row with vlines
        for vline_x, label in vlines_i:
            ax_top.axvline(vline_x, color='dimgray', linestyle='dotted', linewidth=1.4)
            ax_top.annotate(label, xy=(vline_x, 0.4), xytext=(vline_x + 0.01, 0.4),
                            textcoords='data', rotation=90, fontsize=10, color='black', fontweight='bold')

        for j in range(3):
            y = y_list[i][j]
            lower, upper = ci_list[i][j]
            ax_top.plot(x, y, label=labels[i][j], color=colors[j], linestyle=linestyles[j])
            ax_top.fill_between(x, lower, upper, color=colors[j], alpha=0.2)

        ax_top.set_title(titles[i], fontsize=16)
        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, 1)
        ax_top.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        if i == 0:
            ax_top.set_ylabel("Metric Score", fontsize=13)
        if i == 1:
            ax_top.legend(loc="lower center")  # Legend only in left-most top panel
        else:
            ax_top.legend().remove()

        # Bar plot row with vlines
        class0, class1 = freq_list[i]
        for vline_x, label in vlines_i:
            ax_bottom.axvline(vline_x, color='black', linestyle='dotted', linewidth=1.4)
            ax_bottom.annotate(label, xy=(vline_x, 5), xytext=(vline_x + 0.01, 5),
                               textcoords='data', rotation=90, fontsize=10, color='black', fontweight='bold')

        ax_bottom.bar(x, np.log1p(class0), color="orange", width=0.015, alpha=0.9, label="Neg")
        ax_bottom.bar(x, np.log1p(class1), bottom=np.log1p(class0), color="#4c78a8", width=0.015, alpha=0.9, label="Pos")
        ax_bottom.set_xlim(0, 1)
        ax_bottom.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        if i == 0:
            ax_bottom.set_ylabel("log(Count + 1)", fontsize=13)
        if i == 1:
            ax_bottom.legend(loc="upper center")  # Legend only in left-most bottom panel
        else:
            ax_bottom.legend().remove()
        ax_bottom.set_xlabel("Threshold")
        
    plt.tight_layout(h_pad=2.0)

    if save_path_base:
        fig.savefig(f"{save_path_base}.pdf", dpi=600, bbox_inches='tight')
        fig.savefig(f"{save_path_base}.png", dpi=600, bbox_inches='tight')

    return fig, axs

"""
fig, axs = plot_oup_line_and_freq_subplots(
    x=x,
    y_list=y_list,
    ci_list=ci_list,
    labels=labels,
    titles=["PRBC", "FFP", "PLAT"],
    freq_list=class_freq_list,
    vlines=[
        [(0.15, "F-beta=0.15"),(0.52, "Accuracy=0.52"),(0.33, "Kappa=0.33")],
        [(0.17, "F-beta=0.17"), (0.65, "Accuracy=0.65"),(0.22, "Kappa=0.22")],
        [(0.18, "F-beta/Kappa=0.18"),(0.58, "Accuracy=0.58"), (0.18, "")],
    ],
    save_path_base="Figures_Colin/BP_metrics_frequencies"
)
plt.show()
"""

#------------Forest Plot of metrics by time-point-------------#


def plot_forest_from_wide_df(
    df_wide,
    time_col="TimePoint",
    recalib_col="Recalibrated",
    export_folder="Figures_Colin",
    file_prefix="metrics_forest_by_time",
    export_formats=("png", "pdf"),
    symbol_label="★",
    color_time1="steelblue",
    color_recalib="firebrick",
    color_default="black"
):
    """
    Create a multi-panel forest plot from a single wide-format DataFrame.
    Each panel represents a metric, each row a time point with error bars.
    """

    os.makedirs(export_folder, exist_ok=True)

    # Identify metric columns and corresponding CI columns
    metrics = [col for col in df_wide.columns
               if col not in [time_col, recalib_col] and not col.endswith("_CI")]
    n_panels = len(metrics)

    # Establish time order and position mapping
    unique_times = df_wide[time_col].dropna().unique()
    time_order = sorted(unique_times, key=lambda x: int(x.split()[-1]))  # assumes "Time 1", "Time 2", ...
    df_wide[time_col] = pd.Categorical(df_wide[time_col], categories=time_order, ordered=True)
    time_to_y = {tp: i for i, tp in enumerate(time_order)}
    y_positions = list(time_to_y.values())
    y_labels = list(time_to_y.keys())

    # Set up subplots
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        ci_col = f"{metric}_CI"
        for _, row in df_wide.iterrows():
            ypos = time_to_y[row[time_col]]
            lower, upper = row[ci_col]
            median = row[metric]
            err_low = median - lower
            err_high = upper - median

            # Determine color
            if row[recalib_col]:
                color = color_recalib
            elif row[time_col] == "Time 1":
                color = color_time1
            else:
                color = color_default

            # Plot error bars
            ax.errorbar(median, ypos,
                        xerr=[[err_low], [err_high]],
                        fmt='o', color=color, capsize=4)

            # Annotate recalibration symbol
            if row[recalib_col]:
                ax.text(upper + 0.015, ypos, symbol_label,
                        va='center', ha='left', fontsize=10, color=color)

        ax.set_title(metric)
        ax.set_xlim(0, 1.05)
        ax.axvline(0.5, linestyle="--", color="gray", linewidth=0.8)
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5)
        ax.set_xlabel("Median Score with 95% CI")

        # Set y-axis ticks and labels
        ax.set_yticks(y_positions)
        if i == 0:
            ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold', color='black')
            ax.tick_params(axis='y', labelsize=12, labelcolor='black', labelleft=True)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', labelleft=False)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label="Time 1", markerfacecolor=color_time1, markersize=8),
        Line2D([0], [0], marker='o', color='w', label="Recalibrated", markerfacecolor=color_recalib, markersize=8),
        Line2D([0], [0], marker='o', color='w', label="Other", markerfacecolor=color_default, markersize=8)
    ]
    axes[-1].legend(handles=legend_elements, title="Category", loc='upper right')

    # Layout and save
    # Reset leftmost panel labels AFTER layout
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=12, fontweight='bold', color='black')
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)

    for fmt in export_formats:
        path = os.path.join(export_folder, f"{file_prefix}.{fmt}")
        plt.savefig(path, dpi=300, format=fmt, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close()




#---------------------Plotting metrics and thresholds time-series------------------------#


def plot_group_split(df_top, df_sub, group_name, type_labels, custom_colors, val_1="Value", val_2="Value", save_path_base=None):

    df_sub['Group'] = pd.to_datetime(df_sub['Group'], utc=True).dt.tz_convert(None)
    df_top['Group'] = pd.to_datetime(df_top['Group'], utc=True).dt.tz_convert(None)

    # Extract top-row data relevant to the current metric (val_1)
    df_top = df_top[df_top['Product'] == val_2]

    # Use a default secondary metric (e.g., 'Threshold') or find it based on val_1
    secondary_metric = df_top['Threshold'].name if 'Threshold' in df_top.columns else 'Value'

    # Split at "2024-05"
    cutoff = pd.to_datetime("2024-05")
    df_left = df_sub[df_sub['Group'] <= cutoff]
    df_right = df_sub[df_sub['Group'] > cutoff]
    df_top_left = df_top[df_top['Group'] <= cutoff]
    df_top_right = df_top[df_top['Group'] > cutoff]

    # Define x-limits for each panel with safe fallbacks
    def safe_xlim(df):
        if df.empty or df['Group'].isnull().all():
            return None
        return (df['Group'].min(), df['Group'].max())

    left_xlim = safe_xlim(df_left)
    right_xlim = safe_xlim(df_right)

    # OUP style
    sns.set(style="whitegrid", font="Arial", font_scale=1.2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex='col', sharey='row')

    # Top row: secondary metric lineplots (from df_top)
    for ax, df_side, title, xlim in zip(axes[0], [df_top_left, df_top_right], ["Validation", "Test"], [left_xlim, right_xlim]):
        if not df_side.empty:
            sns.lineplot(
                data=df_side,
                x='Group', y=secondary_metric,
                hue='TypeLabel', marker='o',
                palette=custom_colors, ax=ax,
                ci=None  # Disable confidence intervals
            )
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        if xlim:
            ax.set_xlim(xlim)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("")
        ax.set_ylabel(secondary_metric if ax == axes[0, 0] else "")
        ax.set_title(f"{title}: {secondary_metric}", fontweight='bold', fontsize=15)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.9)

    # Shared legend for top row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend().remove()
    axes[0, 1].legend(handles=handles, labels=labels, title='Threshold', loc='upper right', fontsize=10)

    # Bottom row: primary metric lineplots (from df_sub)
    for ax, df_side, title, xlim in zip(axes[1], [df_left, df_right], ["Validation", "Test"], [left_xlim, right_xlim]):
        if not df_side.empty:
            sns.lineplot(
                data=df_side,
                x='Group', y='Value',
                hue='TypeLabel', marker='o',
                palette=custom_colors, ax=ax,
                ci=None  # Disable confidence intervals
            )
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        if xlim:
            ax.set_xlim(xlim)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Month")
        ax.set_ylabel(val_1 if ax == axes[1, 0] else "")
        ax.set_title(f"{title}: {val_1}", fontweight='bold', fontsize=15)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.9)

    # Shared legend for bottom row
    handles, labels = axes[1, 1].get_legend_handles_labels()
    axes[1, 0].legend().remove()
    axes[1, 1].legend(handles=handles, labels=labels, title='Model', loc='lower right', fontsize=10)

    # Super title and layout
    plt.suptitle(f"{group_name}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.5)

    # Save
    if save_path_base:
        os.makedirs(os.path.dirname(save_path_base), exist_ok=True)
        fig.savefig(f"{save_path_base}.pdf", dpi=600, bbox_inches='tight')
        fig.savefig(f"{save_path_base}.png", dpi=600, bbox_inches='tight')
        print(f"Saved: {save_path_base}.pdf / .png")

    plt.close()

"""
for (val_1, val_2), df_sub in monthly_metrics.groupby(['Metric', 'Product']):
    group_name = f"{val_1} for {val_2}"
    save_path_base = f"Figures_OUP/{val_1}_{val_2}".replace(" ", "_")

    plot_group_split(
        df_top=df_thresh,
        df_sub=df_sub,
        group_name=group_name,
        type_labels=type_labels,
        custom_colors=color_dict,
        val_1=val_1,
        val_2=val_2,  # or a custom label if you have one
        save_path_base=save_path_base
    )
"""
    
