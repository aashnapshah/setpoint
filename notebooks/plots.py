import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot styling configuration
PLOT_STYLE = {
    'figure.dpi': 200,
    'font.size': 6,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'axes.labelsize': 6,  
    'legend.fontsize': 6,
    'axes.titlesize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'axes.edgecolor': 'k',
    'axes.linewidth': 0.5,
    'axes.grid': False,
    'axes.prop_cycle': plt.cycler(color=sns.husl_palette(h=.7)),
    'figure.figsize': (2, 2),
    'xtick.major.pad': -3,
    'ytick.major.pad': -3
}
palette = sns.husl_palette(h=.7)

plt.rcParams.update(PLOT_STYLE)
sns.set_theme(style="white", font='Arial', rc=PLOT_STYLE)

def plot_concordance(results, title):
    # get code and columns that end with _concordance
    concordance_cols = [col for col in results.columns if col.endswith('concordance')]
    results = results[['Code'] + concordance_cols + ['Biomarker']]
    results = results.melt(id_vars=['Code', 'Biomarker'], var_name='Set', value_name='concordance')
    results['Set'] = results['Set'].str.split('_').str[0].str.capitalize()
    # plot the code on the x axis and the concordance on the y axis
    fig, ax = plt.subplots(1, 2, figsize=(5, 2), sharey=True)
    df_setpoint = results[results['Biomarker'] == 'setpoint_normalized']
    df_cv = results[results['Biomarker'] == 'cv_normalized']
    sns.barplot(data=df_setpoint, x='Code', y='concordance', hue='Set', ax=ax[0])
    sns.barplot(data=df_cv, x='Code', y='concordance', hue='Set', ax=ax[1])
    ax[0].set_title('Setpoint')
    ax[1].set_title('CV')
    plt.suptitle(title)
    return fig, ax

    
def plot_setpoint_hr(results, title):
    # Create the scatter plot
    results = results.sort_values(by='Code')
    results['Biomarker'] = results['Biomarker'].str.split('_').str[0].str.capitalize()
    results['Code'] = results['Code'] + ' (N=' + results['N'].astype(str) + ')'
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(4, 2))
    # want to plot the HR for setpoint on the left and the HR for cv on the right
    results_setpoint = results[results['Biomarker'] == 'Setpoint']
    results_cv = results[results['Biomarker'] == 'Cv']
    
    sns.scatterplot(data=results_setpoint, x='HR', y='Code', ax=ax[0])
    sns.scatterplot(data=results_cv, x='HR', y='Code', ax=ax[1])
    
    ax[0].errorbar(x=results_setpoint['HR'], y=range(len(results_setpoint)), 
                xerr=[results_setpoint['HR'] - results_setpoint['LOWER CI'], 
                    results_setpoint['UPPER CI'] - results_setpoint['HR']],
                fmt='none', c='gray', alpha=0.5, capsize=3)
    ax[0].set_xlabel('HR')
    ax[0].set_ylabel('')
    ax[0].set_title('Setpoint')
    ax[0].axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax[0].set_xlim(-0.1, 2.2)
    
    ax[1].errorbar(x=results_cv['HR'], y=range(len(results_cv)), 
                xerr=[results_cv['HR'] - results_cv['LOWER CI'], 
                    results_cv['UPPER CI'] - results_cv['HR']],
                fmt='none', c='gray', alpha=0.5, capsize=3)
    ax[1].set_xlabel('HR')
    ax[1].set_ylabel('')
    ax[1].set_title('CV')
    ax[1].axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax[1].set_xlim(0.8, 1.4)
    
    plt.suptitle(title, fontsize=6)
    plt.tight_layout()
    return plt


def plot_interval_hr(results, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure  # Get the figure from the provided axis

    results = results.copy(deep=True)
    results['Biomarker'] = results['Biomarker'].str.split('_').str[2].str.capitalize()
    results['Code'] = results['Code'] + ' (N=' + results['N'].astype(str) + ')'
    results = results.sort_values(by='Code')
    
    palette = sns.husl_palette(h=.7)
    colors = {'Setpoint': 'orange', 'Reference': palette[0]}

    sns.scatterplot(
        data=results, 
        x='HR', 
        y='Code', 
        hue='Biomarker',
        palette=colors,
        s=50,  
        alpha=0.7,
        ax=ax  
    )

    for i, type_name in enumerate(results['Biomarker'].unique()):
        type_df = results[results['Biomarker'] == type_name]
        ax.errorbar(
            x=type_df['HR'], 
            y=type_df['Code'],
            xerr=[type_df['HR'] - type_df['LOWER CI'], 
                  type_df['UPPER CI'] - type_df['HR']],
            fmt='none', 
            c=colors[type_name],
            alpha=0.3,
            capsize=3,
            capthick=1,
            elinewidth=1
        )

    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.set_xlabel('Hazard Ratio (95% CI)')
    ax.set_ylabel('CBC Measure')
    ax.set_title('Hazard Ratio')

    ax.legend().remove()

    plt.xticks()
    plt.yticks()
    plt.tight_layout()

    return fig, ax


def plot_interval_concordance(results, ax=None):
    results = results.copy(deep=True)  # Ensure we don't modify the original dataframe
    results["code"] = results["code"].astype(str)
    results["prediction_time"] = results["prediction_time"].astype(str)
    results["Interval"] = results["biomarker"].astype(str).str.split('_').str[2].str.capitalize()
    results["Time"] = results["prediction_time"] + ' years'

    unique_codes = sorted(results["code"].unique())
    code_mapping = {code: i * 2 for i, code in enumerate(unique_codes)}  # Space out categories
    colors = {'Setpoint': 'orange', 'Reference': palette[0]}

    prediction_times = sorted(results["prediction_time"].unique())
    time_offsets = {time: (i - len(prediction_times) / 2) * 0.5 for i, time in enumerate(prediction_times)}

    results["x_pos"] = results.apply(lambda row: code_mapping[row["code"]] + time_offsets[row["prediction_time"]], axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))  
    else:
        fig = ax.figure  

    # Add error bars
    for type_name in results['Interval'].unique():
        type_df = results[results['Interval'] == type_name]
        ax.errorbar(x=type_df["x_pos"], 
                    y=type_df["c_index"],
                    yerr=[type_df["c_index"] - type_df["ci_lower"], type_df["ci_upper"] - type_df["c_index"]],
                    fmt='none',
                    color=colors[type_name],
                    alpha=0.3,
                    capsize=3,
                    capthick=0.5,
                    linewidth=0.5)
    
    # Scatter plot
    sns.scatterplot(
        data=results,
        x="x_pos",
        y="c_index",
        hue="Interval",
        style="Time",
        alpha=0.8, 
        zorder=10,
        palette=colors.values(),
        ax=ax  # Use the provided ax
    )

    # Labels and title
    ax.set_ylim(0.5, 1)
    ax.set_xlabel("Code")
    ax.set_ylabel("C-Index")

    # Set x-ticks at evenly spaced positions corresponding to each code
    ax.set_xticks([code_mapping[code] for code in unique_codes])
    ax.set_xticklabels(unique_codes)

    # Extract legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    interval_handles = handles[:len(colors)+1]
    interval_labels = labels[:len(colors)+1]
    time_handles = handles[len(colors)+1:]
    time_labels = labels[len(colors)+1:]

    # Create a new legend with two columns
    legend1 = ax.legend(interval_handles, interval_labels, loc='upper left', bbox_to_anchor=(1.02, 1),
                         frameon=False)
    legend2 = ax.legend(time_handles, time_labels, loc='upper left', bbox_to_anchor=(1.02, 0.8),
                         frameon=False)

    # Add both legends
    ax.add_artist(legend1)
    ax.set_title("Concordance Index")
    return fig, ax


def plot_interval_hr_concordance(results, title):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3),  gridspec_kw={'width_ratios': [1, 1.5]})

    # Pass the subplot axes to the functions
    plot_interval_hr(results[0], ax=axes[0])
    plot_interval_concordance(results[1], ax=axes[1])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
