import pandas as pd
import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

# set text size for plots to be larger
plt.rcParams.update({"font.size": 16})

import glob
import os
import tqdm

model_path = os.getcwd() + "/data/2023-04-23/model/"
agent_path = os.getcwd() + "/data/2023-04-23/agent/"
image_path = os.getcwd() + "/model_analysis/2023-04-23/"

os.makedirs(image_path, exist_ok=True)

def set_paths(cwd):
    model_path = cwd + "/data/2023-04-22/model/"
    agent_path = cwd + "/data/2023-04-22/agent/"
    image_path = cwd + "/model_analysis/2023-04-22/"


def load_and_process_file(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_parquet(file_path)

    df = df.reset_index().rename(columns={"index": "Step"})

    # Extract the needed values from the file name
    file_name = os.path.basename(file_path)

    # Create the new column name
    new_col_name = f"{file_name}"

    # change model name to be more readable
    new_col_name = new_col_name.replace("model_seed_", "Seed ")
    new_col_name = new_col_name.replace("agent_seed_", " Seed ")
    new_col_name = new_col_name.replace("_pp_", " PP ")
    new_col_name = new_col_name.replace("_sd", " SD ")
    new_col_name = new_col_name.replace("_ep_", " Ep ")
    new_col_name = new_col_name.replace("_rank_", " Rank ")
    new_col_name = new_col_name.replace("_iter_", " Iter ")
    new_col_name = new_col_name.replace(".parquet", "")

    # Add the new column to the DataFrame
    df["Model"] = new_col_name
    df["file_name"] = file_name

    df["Model"] = df.apply(
        lambda row: f"Seed {row['Seed']} SD{row['Security Density']} PP {row['Private Preference']} EP {row['Episilon']} Th {row['Threshold']}",
        axis=1,
    )

    return df


def get_revo_count(df):
    # Filter the DataFrame based on the 'Revolution' column
    revolutions_df = df[df["Revolution"] == True]

    # Count the unique "models" in the filtered DataFrame
    unique_model_count = revolutions_df["Model"].nunique()
    print(f"{unique_model_count} different 'models' had a revolution.")

def get_combined_count(df):
    # Create a new column "Combined Count" which is the sum of Active Count and Oppo Count
    df["Combined Count"] = df["Active Count"] + df["Oppose Count"]

    # Filter the DataFrame based on the condition where "Combined Count" is greater than 560
    over_half = df[df["Combined Count"] > 560]

    # Count the unique models in the filtered DataFrame
    unique_model_count = over_half["Model"].unique()

    # Create a new column "Over Half" and set it to True if the model is in unique_model_count, False otherwise
    df["Over Half"] = df["Model"].apply(lambda x: True if x in unique_model_count else False)

    print(f"{len(unique_model_count)} models had more than half of the population active or opposing.")

    return df

def revo_count_ep_sd_plot(df, file=None):
    """
    Plot of the number of models that have a revolution by Epsilon and Security Density
    """
    # Filter the DataFrame for Revolution == True
    revolution_true = df[df['Revolution'] == True]['Model'].drop_duplicates()

    # Pull all models that have a revolution
    revolution_true_df = df[df['Model'].isin(revolution_true)]

    # Group the data by Epsilon, Seed, and Model, and count the unique models
    grouped_df = revolution_true_df.groupby(['Epsilon', 'Security Density', 'Model']).size().reset_index(name='Count')

    # Filter the grouped_df to keep only rows with Count >= 1
    filtered_grouped_df = grouped_df[grouped_df['Count'] >= 1]

    # Aggregate the counts by Epsilon and Seed
    agg_df = filtered_grouped_df.groupby(['Epsilon', 'Security Density']).size().reset_index(name='Count')

    # Create a bar plot using Seaborn's barplot
    sns.set(style="whitegrid")
    sns.set_palette("bright")
    sns.set_context("notebook", font_scale=1.5)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=agg_df, x='Epsilon', y='Count', hue='Security Density')

    # Customize the plot
    plt.legend(title="Security Density", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Number of Revolutions by Epsilon and Security Density")
    plt.xlabel("Epsilon")
    plt.ylabel("Count of Models with a Revolution")

    # Save the figure at 300 dpi
    plt.savefig(image_path + file +"revolutions_bar_true_revolutions_epsilon_sd.png", dpi=300, bbox_inches="tight")

    # Display the plot
    plt.show()

def over_half_ep_sd_plot(df, file=None):
    """
    Plot of the number of models that have more than half of the population active or opposing by Epsilon and Security Density
    """
    # Filter the DataFrame for Revolution == True
    combined_count = df[df['Over Half'] == True]['Model'].drop_duplicates()

    # Pull all models that have a revolution
    combined_count_df = df[df['Model'].isin(combined_count)]

    # Group the data by Epsilon, Seed, and Model, and count the unique models
    grouped_df = combined_count_df.groupby(['Epsilon', 'Security Density', 'Model']).size().reset_index(name='Count')

    # Filter the grouped_df to keep only rows with Count >= 112
    filtered_grouped_df = grouped_df[grouped_df['Count'] >= 112]

    # Aggregate the counts by Epsilon and Seed
    agg_df = filtered_grouped_df.groupby(['Epsilon', 'Security Density']).size().reset_index(name='Count')

    # Create a bar plot using Seaborn's barplot
    sns.set(style="whitegrid")
    sns.set_palette("bright")
    sns.set_context("notebook", font_scale=1.5)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=agg_df, x='Epsilon', y='Count', hue='Security Density')

    # Customize the plot
    plt.legend(title="Security Density", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Number of Model Runs with Over 10% Active and Opposed by Epsilon and Security Density")
    plt.xlabel("Epsilon")
    plt.ylabel("Count of Models Over Half")

    # Save the figure at 300 dpi
    plt.savefig(image_path + file + "over_112_bar_epsilon_sd.png", dpi=300, bbox_inches="tight")

    # Display the plot
    plt.show()

def revo_count_ep_th_plot(df, file=None):
    """
    Plot of the number of models that have a revolution by Epsilon and Threshold
    """
    # Filter the DataFrame for Revolution == True
    revolution_true = df[df['Revolution'] == True]['Model'].drop_duplicates()

    # Pull all models that have a revolution
    revolution_true_df = df[df['Model'].isin(revolution_true)]

    # Group the data by Epsilon, Seed, and Model, and count the unique models
    grouped_df = revolution_true_df.groupby(['Threshold', 'Epsilon', 'Model']).size().reset_index(name='Count')

    # Filter the grouped_df to keep only rows with Count >= 1
    filtered_grouped_df = grouped_df[grouped_df['Count'] >= 1]

    # Aggregate the counts by Epsilon and Seed
    agg_df = filtered_grouped_df.groupby(['Threshold', 'Epsilon']).size().reset_index(name='Count')

    # Create a bar plot using Seaborn's barplot
    sns.set(style="whitegrid")
    sns.set_palette("coolwarm")
    sns.set_context("notebook", font_scale=1.5)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=agg_df, x='Threshold', y='Count', hue='Epsilon')

    # Customize the plot
    plt.legend(title="Epsilon", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Number of Revolutions by Threshold and Epsilon")
    plt.xlabel("Threshold")
    plt.ylabel("Count of Models with a Revolution")

    # Save the figure at 300 dpi
    plt.savefig(image_path + file + "revolutions_bar_true_revolutions_threshold_epsilon.png", dpi=300, bbox_inches="tight")

    # Display the plot
    plt.show()

def over_half_ep_th_plot(df, file=None):
    """
    Plot of the number of models that have more than half of the population active or opposing by Epsilon and Threshold
    """
    # Filter the DataFrame for Over Half
    combined_count = df[df['Over Half'] == True]['Model'].drop_duplicates()

    # Pull all models that have a revolution
    combined_count_df = df[df['Model'].isin(combined_count)]

    # Group the data by Epsilon, Seed, and Model, and count the unique models
    grouped_df = combined_count_df.groupby(['Threshold', 'Epsilon', 'Model']).size().reset_index(name='Count')

    # Filter the grouped_df to keep only rows with Count >= 1
    filtered_grouped_df = grouped_df[grouped_df['Count'] >= 1]

    # Aggregate the counts by Epsilon and Seed
    agg_df = filtered_grouped_df.groupby(['Threshold', 'Epsilon']).size().reset_index(name='Count')

    # Create a bar plot using Seaborn's barplot
    sns.set(style="whitegrid")
    sns.set_palette("coolwarm")
    sns.set_context("notebook", font_scale=1.5)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=agg_df, x='Threshold', y='Count', hue='Epsilon')

    # Customize the plot
    plt.legend(title="Epsilon", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Number of Model Runs with Over 10% Active and Opposed by Threshold and Epsilon")
    plt.xlabel("Threshold")
    plt.ylabel("Count of Models Over Half")

    # Save the figure at 300 dpi
    plt.savefig(image_path + file + "over_112_bar_threshold_epsilon.png", dpi=300, bbox_inches="tight")

    # Display the plot
    plt.show()

def revo_count_sd_th_plot(df, file=None):
    """
    Plot of the number of models that have a revolution by Security Density and Threshold
    """
    # Filter the DataFrame for Revolution == True
    revolution_true = df[df['Revolution'] == True]['Model'].drop_duplicates()

    # Pull all models that have a revolution
    revolution_true_df = df[df['Model'].isin(revolution_true)]

    # Group the data by Epsilon, Seed, and Model, and count the unique models
    grouped_df = revolution_true_df.groupby(['Threshold', 'Security Density', 'Model']).size().reset_index(name='Count')

    # Filter the grouped_df to keep only rows with Count >= 1
    filtered_grouped_df = grouped_df[grouped_df['Count'] >= 1]

    # Aggregate the counts by Epsilon and Seed
    agg_df = filtered_grouped_df.groupby(['Threshold', 'Security Density']).size().reset_index(name='Count')

    # Create a bar plot using Seaborn's barplot
    sns.set(style="whitegrid")
    sns.set_palette("bright")
    sns.set_context("notebook", font_scale=1.5)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=agg_df, x='Threshold', y='Count', hue='Security Density')

    # Customize the plot
    plt.legend(title="Security Density", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Number of Revolutions by Threshold and Security Density")
    plt.xlabel("Threshold")
    plt.ylabel("Count of Models with a Revolution")

    # Save the figure at 300 dpi
    plt.savefig(image_path + file + "revolutions_bar_true_revolutions_threshold_sd.png", dpi=300, bbox_inches="tight")

    # Display the plot
    plt.show()

def over_half_sd_th_plot(df, file=None):
    """
    Plot of the number of models that have more than half of the population active or opposing by Security Density and Threshold
    """
    # Filter the DataFrame for Over Half == True
    combined_count = df[df['Over Half'] == True]['Model'].drop_duplicates()

    # Pull all models that have a revolution
    combined_count_df = df[df['Model'].isin(combined_count)]

    # Group the data by Epsilon, Seed, and Model, and count the unique models
    grouped_df = combined_count_df.groupby(['Threshold', 'Security Density', 'Model']).size().reset_index(name='Count')

    # Filter the grouped_df to keep only rows with Count >= 1
    filtered_grouped_df = grouped_df[grouped_df['Count'] >= 1]

    # Aggregate the counts by Epsilon and Seed
    agg_df = filtered_grouped_df.groupby(['Threshold', 'Security Density']).size().reset_index(name='Count')

    # Create a bar plot using Seaborn's barplot
    sns.set(style="whitegrid")
    sns.set_palette("bright")
    sns.set_context("notebook", font_scale=1.5)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=agg_df, x='Threshold', y='Count', hue='Security Density')

    # Customize the plot
    plt.legend(title="Security Density", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Number of Model Runs with Over 10% Active and Opposed by Threshold and Security Density")
    plt.xlabel("Threshold")
    plt.ylabel("Count of Models Over Half")

    # Save the figure at 300 dpi
    plt.savefig(image_path + file + "over_112_bar_threshold_sd.png", dpi=300, bbox_inches="tight")

    # Display the plot
    plt.show()

def active_count_ep_plot(df, file=None):
    """
    Plot of actives with error bars by epsilon
    """
    # Set the style, color palette, and font size for the plots
    sns.set(style="whitegrid")
    sns.set_palette("deep")
    sns.set_context("notebook", font_scale=1.5)

    # Create the lineplot using Seaborn's lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Active Count", hue="Epsilon")

    # Customize the plot
    plt.legend(title="Episilon", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Active Count per Step by Epsilon")
    plt.xlabel("Step")
    plt.ylabel("Active Count")

    # Save the figure at 300 dpi
    plt.savefig(
        image_path + file + "line_active_count_epsilon.png", dpi=300, bbox_inches="tight"
    )

    # Display the plot
    plt.show()

def active_count_th_plot(df, file=None):
    """
    Plot of actives with error bars by threshold
    """
    # Set the style, color palette, and font size for the plots
    sns.set(style="whitegrid")
    sns.set_palette("deep")
    sns.set_context("notebook", font_scale=1.5)

    # Create the lineplot using Seaborn's lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Active Count", hue="Threshold")

    # Customize the plot
    plt.legend(title="Threshold", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Active Count per Step by Threshold")
    plt.xlabel("Step")
    plt.ylabel("Active Count")

    # Save the figure at 300 dpi
    plt.savefig(
        image_path + file + "line_active_count_threshold.png", dpi=300, bbox_inches="tight"
    )

    # Display the plot
    plt.show()

def active_count_sd_plot(df, file=None):
    """
    Plot of actives with error bars by security density
    """
    # Set the style, color palette, and font size for the plots
    sns.set(style="whitegrid")
    sns.set_palette("deep")
    sns.set_context("notebook", font_scale=1.5)

    # Create the lineplot using Seaborn's lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Active Count", hue="Security Density")

    # Customize the plot
    plt.legend(title="Security Density", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Active Count per Step by Security Density")
    plt.xlabel("Step")
    plt.ylabel("Active Count")

    # Save the figure at 300 dpi
    plt.savefig(
        image_path + file + "line_active_count_security_density.png", dpi=300, bbox_inches="tight"
    )

    # Display the plot
    plt.show()

def active_count_ep_sd_plot(df, file=None):
    """
    Plot of actives with error bars by epsilon and security density
    """
    # Set the style, color palette, and font size for the plots
    sns.set(style="whitegrid")
    sns.set_palette("deep")
    sns.set_context("notebook", font_scale=1.5)

    # Create the lineplot using Seaborn's lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Active Count", hue="Epsilon", style="Security Density")

    # Customize the plot
    plt.legend(title="Episilon", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Active Count per Step by Epsilon and Security Density")
    plt.xlabel("Step")
    plt.ylabel("Active Count")

    # Save the figure at 300 dpi
    plt.savefig(
        image_path + file + "line_active_count_epsilon_sd.png", dpi=300, bbox_inches="tight"
    )

    # Display the plot
    plt.show()

def over_half_active_count_ep_plot(df, file=None):
    """
    Plot of models with over half active with error bars by epsilon
    """
    # Filter the DataFrame for Over Half
    combined_count = df[df['Over Half'] == True]['Model'].drop_duplicates()

    # Pull all models with over half active
    combined_count_df = df[df['Model'].isin(combined_count)]

    # Set the style, color palette, and font size for the plots
    sns.set(style="whitegrid")
    sns.set_palette("deep")
    sns.set_context("notebook", font_scale=1.5)

    # Create the lineplot using Seaborn's lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_count_df, x="Step", y="Active Count", hue="Epsilon")

    # Customize the plot
    plt.legend(title="Episilon", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Active Count per Step by Epsilon where Over Half Active")
    plt.xlabel("Step")
    plt.ylabel("Active Count")

    # Save the figure at 300 dpi
    plt.savefig(
        image_path + file + "line_over_half_active_count_epsilon.png", dpi=300, bbox_inches="tight"
    )

    # Display the plot
    plt.show()

def over_half_active_count_th_plot(df, file=None):
    """
    Plot of models with over half active with error bars by threshold
    """
    # Filter the DataFrame for Over Half
    combined_count = df[df['Over Half'] == True]['Model'].drop_duplicates()

    # Pull all models with over half active
    combined_count_df = df[df['Model'].isin(combined_count)]

    # Set the style, color palette, and font size for the plots
    sns.set(style="whitegrid")
    sns.set_palette("deep")
    sns.set_context("notebook", font_scale=1.5)

    # Create the lineplot using Seaborn's lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_count_df, x="Step", y="Active Count", hue="Threshold")

    # Customize the plot
    plt.legend(title="Threshold", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Active Count per Step by Threshold where Over Half Active")
    plt.xlabel("Step")
    plt.ylabel("Active Count")

    # Save the figure at 300 dpi
    plt.savefig(
        image_path + file + "line_over_half_active_count_threshold.png", dpi=300, bbox_inches="tight"
    )

    # Display the plot
    plt.show()

def over_half_active_count_sd_plot(df, file=None):
    """
    Plot of models with over half active with error bars by security density
    """
    # Filter the DataFrame for Over Half
    combined_count = df[df['Over Half'] == True]['Model'].drop_duplicates()

    # Pull all models with over half active
    combined_count_df = df[df['Model'].isin(combined_count)]

    # Set the style, color palette, and font size for the plots
    sns.set(style="whitegrid")
    sns.set_palette("deep")
    sns.set_context("notebook", font_scale=1.5)

    # Create the lineplot using Seaborn's lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_count_df, x="Step", y="Active Count", hue="Security Density")

    # Customize the plot
    plt.legend(title="Security Density", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Active Count per Step by Security Density where Over Half Active")
    plt.xlabel("Step")
    plt.ylabel("Active Count")

    # Save the figure at 300 dpi
    plt.savefig(
        image_path + file + "line_over_half_active_count_security_density.png", dpi=300, bbox_inches="tight"
    )

    # Display the plot
    plt.show()

def check_for_ratio_fluctuations(df, lowerbound=0.10, upperbound=0.20):
    # Find the models with a ratio of actives to total citizen agents above 20%
    df['ActiveRatio'] = df['Active Count'] / df['Citizen Count']

    # Initialize an empty set to store the models meeting the criteria
    fluctuating_models = set()

    # Initialize a dictionary to store the state of each model
    model_states = {}

    # Loop through the DataFrame rows
    for model in df['Model'].unique():
        for index, row in df[df['Model'] == model].iterrows():
            model = row['Model']
            step = row['Step']
            active_ratio = row['ActiveRatio']

            if model not in model_states:
                model_states[model] = {f"first_above_{upperbound}": None, f"below_{lowerbound}": False}

            if active_ratio > upperbound:
                if model_states[model][f"first_above_{upperbound}"] is None:
                    model_states[model][f"first_above_{upperbound}"] = step
                elif model_states[model][f"below_{lowerbound}"]:
                    fluctuating_models.add(model)
                    break
            elif active_ratio < lowerbound and model_states[model][f"first_above_{upperbound}"] is not None:
                model_states[model][f"below_{lowerbound}"] = True

    return fluctuating_models

def histogram_over_112(df, file=None):
    # Filter the final_df DataFrame for rows with count of Active > 112
    filtered_df = df[df["Active Count"] > 112]

    # Identify the unique models in the filtered DataFrame
    unique_models = filtered_df["Model"].unique()

    # Filter the final_df DataFrame for the specified models
    filtered_df = df[df["Model"].isin(unique_models)]

    # Group the dataframe by the 'model' column and find the maximum speed of spread for each group
    max_speed_df = filtered_df.groupby('Model')['Speed of Spread'].max().reset_index()

    # Merge the max_speed_df with the epsilon values
    max_speed_epsilon_df = max_speed_df.merge(filtered_df[['Model', 'Epsilon']].drop_duplicates(), on='Model')

    # Set the style for the plot
    sns.set(style='whitegrid')

    # Map Epsilon values to colors in a Red-Blue spectrum
    unique_epsilons = np.sort(max_speed_epsilon_df['Epsilon'].unique())
    norm = plt.Normalize(vmin=unique_epsilons.min(), vmax=unique_epsilons.max())
    cmap = plt.get_cmap('coolwarm')
    epsilon_colors = {epsilon: cmap(norm(epsilon)) for epsilon in unique_epsilons}

    # Create a custom legend using Epsilon values and colors
    legend_elements = [plt.Line2D([0], [0], color=epsilon_colors[epsilon], lw=4, label=f'Epsilon: {epsilon}') for epsilon in unique_epsilons]

    # Create the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=max_speed_epsilon_df, x='Speed of Spread', hue='Epsilon', palette=epsilon_colors, bins=20, kde=True)

    # Customize the plot
    plt.legend(handles=legend_elements, title='Epsilon', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Histogram of Maximum Speed of Spread by Epsilon where Active Count > 112')
    plt.xlabel('Maximum Speed of Spread')
    plt.ylabel('Frequency')

    # Save the figure at 300 dpi
    plt.savefig(image_path + file + 'histogram_max_speed_spread_over_112.png', dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

def histogram_over_half(df, file=None):
    # Filter the final_df DataFrame for rows with over Half Active
    filtered_df = df[df["Over Half"] == True]

    # Identify the unique models in the filtered DataFrame
    unique_models = filtered_df["Model"].unique()

    # Filter the final_df DataFrame for the specified models
    filtered_df = df[df["Model"].isin(unique_models)]

    # Group the dataframe by the 'model' column and find the maximum speed of spread for each group
    max_speed_df = filtered_df.groupby('Model')['Speed of Spread'].max().reset_index()

    # Merge the max_speed_df with the epsilon values
    max_speed_epsilon_df = max_speed_df.merge(filtered_df[['Model', 'Epsilon']].drop_duplicates(), on='Model')

    # Set the style for the plot
    sns.set(style='whitegrid')

    # Map Epsilon values to colors in a Red-Blue spectrum
    unique_epsilons = np.sort(max_speed_epsilon_df['Epsilon'].unique())
    norm = plt.Normalize(vmin=unique_epsilons.min(), vmax=unique_epsilons.max())
    cmap = plt.get_cmap('coolwarm')
    epsilon_colors = {epsilon: cmap(norm(epsilon)) for epsilon in unique_epsilons}

    # Create a custom legend using Epsilon values and colors
    legend_elements = [plt.Line2D([0], [0], color=epsilon_colors[epsilon], lw=4, label=f'Epsilon: {epsilon}') for epsilon in unique_epsilons]

    # Create the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=max_speed_epsilon_df, x='Speed of Spread', hue='Epsilon', palette=epsilon_colors, bins=20, kde=True)

    # Customize the plot
    plt.legend(handles=legend_elements, title='Epsilon', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Histogram of Maximum Speed of Spread by Epsilon Where Over Half Active')
    plt.xlabel('Maximum Speed of Spread')
    plt.ylabel('Frequency')

    # Save the figure at 300 dpi
    plt.savefig(image_path + file + 'histogram_max_speed_spread_over_half.png', dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

def count_over_half_private_preference(df, file=None):

    # Filter the DataFrame for Revolution == True
    filtered_df = df[df["Over Half"] == True]['Model'].drop_duplicates()

    # Pull all models that have a revolution
    over_half_df = df[df['Model'].isin(filtered_df)]

    # Group the data by Epsilon, Seed, and Model, and count the unique models
    grouped_df = over_half_df.groupby(['Private Preference', 'Epsilon', 'Model']).size().reset_index(name='Count')

    # Filter the grouped_df to keep only rows with Count >= 1
    filtered_grouped_df = grouped_df[grouped_df['Count'] >= 1]

    # Aggregate the counts by Epsilon and Seed
    agg_df = filtered_grouped_df.groupby(['Private Preference', 'Epsilon']).size().reset_index(name='Count')

    # Set the style for the plot
    sns.set(style='whitegrid')

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=agg_df, x='Private Preference', y='Count', hue='Epsilon', palette='coolwarm')

    # Customize the plot
    plt.title('Count of Models that Reached Over Half for Different Private Preferences')
    plt.xlabel('Private Preference')
    plt.ylabel('Count of Models')

    # Save the figure at 300 dpi
    if file:
        plt.savefig(image_path + file + 'pri_pref_count_models_over_half.png', dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()


def histogram_pri_pref(model_1, model_2, df):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Filter the agent_df for each model at step 0
    agent_df_1 = df[(df['Model'] == model_1) & (df['Step'] == 1)]
    agent_df_2 = df[(df['Model'] == model_2) & (df['Step'] == 1)]

    # Create the overlay plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=agent_df_1, x='private_preference', kde=True, color='blue', label='Seed 1030', alpha=0.5, ax=ax)
    sns.histplot(data=agent_df_2, x='private_preference', kde=True, color='red', label='Seed 1031', alpha=0.5, ax=ax)

    # Customize the plot
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('Step 0 Private Preference for Seed 1030 and Seed 1031')
    ax.set_xlabel('Private Preference')
    ax.set_ylabel('Frequency')

    # Create the inset plot
    ax_inset = inset_axes(ax, width='30%', height='30%', loc='upper right')
    sns.histplot(data=agent_df_1, x='epsilon', kde=True, color='blue', label='Seed 1030', alpha=0.5, ax=ax_inset)
    sns.histplot(data=agent_df_2, x='epsilon', kde=True, color='red', label='Seed 1031', alpha=0.5, ax=ax_inset)
    ax_inset.set_xlabel('Epsilon')
    ax_inset.set_ylabel('Frequency')
    ax_inset.legend().remove()

    # Save the figure at 300 dpi
    plt.savefig(image_path + 'overlay_pp_inset.png', dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

def fluctuate_active_plot(df, fluctuating_models, file=None):
    """
    Plot of actives no error bars by epsilon
    """
    # Filter the final_df DataFrame for the specified models
    filtered_df = df[df["Model"].isin(list(fluctuating_models))]
    
    # Set the style, color palette, and font size for the plots
    sns.set(style="whitegrid")
    sns.set_palette("deep")
    sns.set_context("notebook", font_scale=1.5)

    # Create the lineplot using Seaborn's lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=filtered_df, x="Step", y="Active Count", hue="Epsilon", errorbar=None)

    # Customize the plot
    plt.legend(title="Episilon", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Active Count per Step by Epsilon for Fluctuating Models")
    plt.xlabel("Step")
    plt.ylabel("Active Count")

    # Save the figure at 300 dpi
    plt.savefig(
        image_path + file + "line_active_count_epsilon.png", dpi=300, bbox_inches="tight"
    )

    # Display the plot
    plt.show()