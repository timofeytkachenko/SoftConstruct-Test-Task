import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import math
from collections import Counter
import weightedstats as ws


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


class Med_couple:
    def __init__(self, data):
        self.data = np.sort(data, axis=None)[::-1]  # sorted decreasing
        self.med = np.median(self.data)
        self.scale = 2 * np.amax(np.absolute(self.data))
        self.Zplus = [(x - self.med) / self.scale for x in self.data if x >= self.med]
        self.Zminus = [(x - self.med) / self.scale for x in self.data if x <= self.med]
        self.p = len(self.Zplus)
        self.q = len(self.Zminus)

    def H(self, i, j):
        a = self.Zplus[i]
        b = self.Zminus[j]

        if a == b:
            return np.sign(self.p - 1 - i - j)
        else:
            return (a + b) / (a - b)

    def greater_h(self, u):

        P = [0] * self.p

        j = 0

        for i in range(self.p - 1, -1, -1):
            while j < self.q and self.H(i, j) > u:
                j += 1
            P[i] = j - 1
        return P

    def less_h(self, u):

        Q = [0] * self.p

        j = self.q - 1

        for i in range(self.p):
            while j >= 0 and self.H(i, j) < u:
                j = j - 1
            Q[i] = j + 1

        return Q

    # Kth pair algorithm (Johnson & Mizoguchi)
    def kth_pair_algorithm(self):
        L = [0] * self.p
        R = [self.q - 1] * self.p

        Ltotal = 0

        Rtotal = self.p * self.q

        medcouple_index = math.floor(Rtotal / 2)

        while Rtotal - Ltotal > self.p:

            middle_idx = [i for i in range(self.p) if L[i] <= R[i]]
            row_medians = [self.H(i, math.floor((L[i] + R[i]) / 2)) for i in middle_idx]

            weight = [R[i] - L[i] + 1 for i in middle_idx]

            WM = ws.weighted_median(row_medians, weights=weight)

            P = self.greater_h(WM)

            Q = self.less_h(WM)

            Ptotal = np.sum(P) + len(P)
            Qtotal = np.sum(Q)

            if medcouple_index <= Ptotal - 1:
                R = P
                Rtotal = Ptotal
            else:
                if medcouple_index > Qtotal - 1:
                    L = Q
                    Ltotal = Qtotal
                else:
                    return WM
        remaining = np.array([])

        for i in range(self.p):
            for j in range(L[i], R[i] + 1):
                remaining = np.append(remaining, self.H(i, j))

        find_index = medcouple_index - Ltotal

        k_minimum_element = remaining[
            np.argpartition(remaining, find_index)
        ]  # K-element algothrm

        return k_minimum_element[find_index]


def detection_outlier(n, df):
    outlier_indices = []

    for col in df.columns:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        medcouple = Med_couple(np.array(df[col])).kth_pair_algorithm()
        if medcouple >= 0:
            outlier_list_col = df[
                (df[col] < Q1 - outlier_step * math.exp(-3.5 * medcouple))
                | (df[col] > Q3 + outlier_step * math.exp(4 * medcouple))
            ].index
        else:
            outlier_list_col = df[
                (df[col] < Q1 - outlier_step * math.exp(-4 * medcouple))
                | (df[col] > Q3 + outlier_step * math.exp(3.5 * medcouple))
            ].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers
