import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import math


def plot_feature_distributions(data):
    data_columns = data.columns
    num_features = len(data_columns)
    num_cols = 3
    num_rows = (
        num_features + num_cols - 1
    ) // num_cols  # Calculate the number of rows needed

    plt.figure(figsize=(20, 5 * num_rows))
    for i, column in enumerate(data_columns):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.kdeplot(data[column], color="blue", fill=True)
        plt.title(column)

    plt.tight_layout()
    plt.show()


def plot_feature_correlations(data):
    correlation_mat = data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_mat, annot=True, cmap="coolwarm")
    plt.show()


def create_interactive_pie_charts(df, num_bins=10):
    for column in df.columns:
        feature = df[column]

        # Determine the minimum and maximum values for bin creation
        min_val = (
            math.floor(feature.min() * 10) / 10
            if feature.min() >= 0 and feature.max() <= 1
            else feature.min()
        )
        max_val = (
            math.ceil(feature.max() * 10) / 10
            if feature.min() >= 0 and feature.max() <= 1
            else feature.max()
        )

        # Create bins
        bins = np.linspace(min_val, max_val, num_bins + 1)

        # Group values into bins
        binned_data = pd.cut(feature, bins, right=True, include_lowest=True)

        # Count the number of values in each bin
        bin_counts = binned_data.value_counts(sort=False).reset_index()
        bin_counts.columns = ["bin", "count"]

        # Convert Interval objects to strings
        bin_counts["bin"] = bin_counts["bin"].astype(str)

        # Create the pie chart
        fig = px.pie(bin_counts, values="count", names="bin", title=f"{column}")

        # Hide the legend
        fig.update_layout(showlegend=False)

        # Show the chart
        fig.show()