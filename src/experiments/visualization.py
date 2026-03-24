#!/usr/bin/env python3
"""
Visualization Module for Case Align Correlation Analysis

This module contains all plotting and visualization functions for analyzing
correlations between Case Align and traditional robustness metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from typing import List, Optional

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


def plot_correlation_matrix(df: pd.DataFrame, metrics: List[str], title: str = None) -> None:
    """Plot correlation heatmap for selected metrics."""
    # Compute correlation matrix
    correlation_matrix = df[metrics].corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(
        correlation_matrix, 
        mask=mask,
        annot=True, 
        cmap='RdBu_r', 
        center=0,
        square=True, 
        linewidths=0.5,
        fmt='.3f',
        ax=ax
    )
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Correlation Matrix: Case Align vs Traditional Robustness Metrics')
    
    plt.tight_layout()
    plt.show()


def plot_correlation_scatterplots(df: pd.DataFrame, case_align_cols: List[str], traditional_cols: List[str]) -> None:
    """Create scatter plots for correlations between case align and traditional metrics."""
    n_plots = len(case_align_cols) * len(traditional_cols)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.ravel() if n_rows > 1 else axes
    
    plot_idx = 0
    for ca_col in case_align_cols:
        for trad_col in traditional_cols:
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            # Remove infinite values for plotting
            mask = np.isfinite(df[ca_col]) & np.isfinite(df[trad_col])
            if mask.sum() < 3:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{ca_col} vs {trad_col}")
                plot_idx += 1
                continue
                
            x = df[trad_col][mask]
            y = df[ca_col][mask]
            
            # Scatter plot colored by class
            if 'class' in df.columns:
                for class_val in df['class'].unique():
                    class_mask = mask & (df['class'] == class_val)
                    if class_mask.sum() > 0:
                        ax.scatter(
                            df[trad_col][class_mask], 
                            df[ca_col][class_mask], 
                            alpha=0.6, 
                            label=f'Class {class_val}',
                            s=40
                        )
            else:
                ax.scatter(x, y, alpha=0.6, s=40)
            
            # Add trend line
            if len(x) > 1 and len(y) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
            
            # Compute correlation for title
            try:
                corr_r, _ = pearsonr(x, y)
                ax.set_title(f"{ca_col.replace('case_align_', '')} vs {trad_col}\nr={corr_r:.3f}", fontsize=10)
            except:
                ax.set_title(f"{ca_col.replace('case_align_', '')} vs {trad_col}", fontsize=10)
            
            ax.set_xlabel(trad_col.replace('_', ' ').title())
            ax.set_ylabel(ca_col.replace('case_align_', '').replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            
            if plot_idx == 0 and 'class' in df.columns:
                ax.legend()
            
            plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_three_way_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str, 
                          x_title: str = None, y_title: str = None, color_title: str = None) -> None:
    """Create a scatter plot with third variable as color."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check for valid data
    mask = (np.isfinite(df[x_col]) & np.isfinite(df[y_col]) & np.isfinite(df[color_col]))
    
    if mask.sum() == 0:
        ax.text(0.5, 0.5, 'No valid data for visualization', ha='center', va='center', transform=ax.transAxes)
        return
    
    scatter = ax.scatter(
        df[x_col][mask], 
        df[y_col][mask],
        c=df[color_col][mask], 
        cmap='viridis', 
        alpha=0.7, 
        s=50
    )
    
    plt.colorbar(scatter, label=color_title or color_col.replace('_', ' ').title())
    
    ax.set_xlabel(x_title or x_col.replace('_', ' ').title())
    ax.set_ylabel(y_title or y_col.replace('_', ' ').title())
    ax.grid(True, alpha=0.3)
    
    # Add correlation in title
    try:
        corr_r, _ = pearsonr(df[x_col][mask], df[y_col][mask])
        ax.set_title(f'Three-way Analysis (r={corr_r:.3f})')
    except:
        ax.set_title('Three-way Analysis')
    
    plt.tight_layout()
    plt.show()


def plot_distributions(df: pd.DataFrame, metrics: List[str]) -> None:
    """Plot distribution histograms for metrics."""
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.ravel() if n_rows > 1 else axes
    
    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Remove infinite values
        finite_data = df[metric][np.isfinite(df[metric])]
        
        if len(finite_data) > 0:
            ax.hist(finite_data, bins=20, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_title(metric.replace('case_align_', '').replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add summary stats
            mean_val = finite_data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, 
                      label=f'Mean: {mean_val:.3f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No finite data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric)
    
    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_class_comparison(df: pd.DataFrame, metrics: List[str]) -> None:
    """Plot box plots comparing metrics by class."""
    if 'class' not in df.columns:
        print("No class column found for comparison")
        return
    
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)  
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.ravel() if n_rows > 1 else axes
    
    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Prepare data for boxplot
        finite_mask = np.isfinite(df[metric])
        plot_data = df[finite_mask]
        
        if len(plot_data) > 0 and len(plot_data['class'].unique()) > 1:
            sns.boxplot(data=plot_data, x='class', y=metric, ax=ax)
            ax.set_title(metric.replace('case_align_', '').replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            
            # Add statistical test result if binary classification
            classes = plot_data['class'].unique()
            if len(classes) == 2:
                try:
                    class0_data = plot_data[plot_data['class'] == classes[0]][metric]
                    class1_data = plot_data[plot_data['class'] == classes[1]][metric]
                    if len(class0_data) > 2 and len(class1_data) > 2:
                        _, p_value = mannwhitneyu(class0_data, class1_data)
                        ax.set_title(f"{metric.replace('case_align_', '').replace('_', ' ').title()}\n(p={p_value:.3f})")
                except:
                    pass
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor class comparison', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric.replace('case_align_', '').replace('_', ' ').title())
    
    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_summary(corr_results_df: pd.DataFrame) -> None:
    """Plot a summary of correlation results."""
    if corr_results_df.empty:
        print("No correlation results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Pearson correlations
    x_pos = np.arange(len(corr_results_df))
    bars1 = ax1.bar(x_pos, corr_results_df['pearson_r'], 
                    color=['red' if r < 0 else 'blue' for r in corr_results_df['pearson_r']],
                    alpha=0.7)
    
    ax1.set_title('Pearson Correlations')
    ax1.set_xlabel('Metric Pairs')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    # Add correlation values on bars
    for i, (bar, val) in enumerate(zip(bars1, corr_results_df['pearson_r'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.05),
                f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Set x-tick labels
    labels = [f"{row['case_align_metric']}\nvs\n{row['traditional_metric']}" 
              for _, row in corr_results_df.iterrows()]
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    
    # Plot 2: Spearman correlations  
    bars2 = ax2.bar(x_pos, corr_results_df['spearman_r'], 
                    color=['red' if r < 0 else 'blue' for r in corr_results_df['spearman_r']],
                    alpha=0.7)
    
    ax2.set_title('Spearman Correlations')
    ax2.set_xlabel('Metric Pairs')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Add correlation values on bars
    for i, (bar, val) in enumerate(zip(bars2, corr_results_df['spearman_r'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.05),
                f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


def plot_sensitivity_diagnosis(df: pd.DataFrame) -> None:
    """Plot diagnostic information for sensitivity analysis."""
    if 'captum_sensitivity' not in df.columns:
        print("No sensitivity data found")
        return
        
    sens_data = df['captum_sensitivity'][np.isfinite(df['captum_sensitivity'])]
    
    if len(sens_data) == 0:
        print("No valid sensitivity data")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Sensitivity distribution
    ax1.hist(sens_data, bins=30, alpha=0.7, edgecolor='black')
    ax1.set_title('Sensitivity Distribution')
    ax1.set_xlabel('Sensitivity Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_sens = sens_data.mean()
    std_sens = sens_data.std()
    ax1.axvline(mean_sens, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_sens:.5f}')
    ax1.axvline(mean_sens - std_sens, color='orange', linestyle=':', alpha=0.7, label=f'±1 STD')
    ax1.axvline(mean_sens + std_sens, color='orange', linestyle=':', alpha=0.7)
    ax1.legend()
    
    # Plot 2: Sensitivity variance analysis
    variance = sens_data.var()
    range_val = sens_data.max() - sens_data.min()
    
    metrics = ['Variance', 'Range', 'Std Dev']
    values = [variance, range_val, std_sens]
    
    bars = ax2.bar(metrics, values, color=['blue', 'green', 'orange'], alpha=0.7)
    ax2.set_title('Sensitivity Variability Metrics')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{val:.6f}', ha='center', va='bottom')
    
    # Add warning if variance is very low
    if variance < 1e-6:
        ax2.text(0.5, 0.8, '⚠ WARNING: Very low variance!\nConsider increasing noise level', 
                transform=ax2.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def create_analysis_summary_plot(df: pd.DataFrame, corr_results_df: pd.DataFrame) -> None:
    """Create a comprehensive summary visualization."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Dataset overview
    ax1 = fig.add_subplot(gs[0, 0])
    if 'class' in df.columns:
        class_counts = df['class'].value_counts()
        ax1.pie(class_counts.values, labels=[f'Class {i}' for i in class_counts.index], 
               autopct='%1.1f%%', startangle=90)
        ax1.set_title('Class Distribution')
    
    # 2. Correlation summary
    ax2 = fig.add_subplot(gs[0, 1:])
    if not corr_results_df.empty:
        x_pos = np.arange(len(corr_results_df))
        width = 0.35
        
        ax2.bar(x_pos - width/2, corr_results_df['pearson_r'], width, 
               label='Pearson', alpha=0.7)
        ax2.bar(x_pos + width/2, corr_results_df['spearman_r'], width, 
               label='Spearman', alpha=0.7)
        
        ax2.set_title('Correlation Comparison')
        ax2.set_xlabel('Metric Pairs')  
        ax2.set_ylabel('Correlation Coefficient')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        labels = [f"{row['case_align_metric']}\nvs\n{row['traditional_metric']}" 
                 for _, row in corr_results_df.iterrows()]
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
    
    # 3. Metric ranges
    ax3 = fig.add_subplot(gs[1, :])
    metrics = ['case_align_S_plus', 'case_align_R_bounded', 'captum_sensitivity', 'knn_similarity_robustness']
    available_metrics = [m for m in metrics if m in df.columns]
    
    data_for_box = []
    labels_for_box = []
    
    for metric in available_metrics:
        finite_data = df[metric][np.isfinite(df[metric])]
        if len(finite_data) > 0:
            data_for_box.append(finite_data)
            labels_for_box.append(metric.replace('case_align_', '').replace('_', ' ').title())
    
    if data_for_box:
        ax3.boxplot(data_for_box, labels=labels_for_box)
        ax3.set_title('Metric Value Distributions')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Data quality indicators
    ax4 = fig.add_subplot(gs[2, :])
    quality_metrics = []
    quality_values = []
    
    for metric in available_metrics:
        finite_count = np.isfinite(df[metric]).sum()
        total_count = len(df[metric])
        finite_pct = (finite_count / total_count) * 100
        
        quality_metrics.append(metric.replace('case_align_', '').replace('_', ' ').title())
        quality_values.append(finite_pct)
    
    if quality_metrics:
        bars = ax4.bar(quality_metrics, quality_values, color='skyblue', alpha=0.7)
        ax4.set_title('Data Quality (% Finite Values)')
        ax4.set_ylabel('Percentage')
        ax4.set_ylim(0, 105)
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Add percentage labels on bars
        for bar, val in zip(bars, quality_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}%', ha='center', va='bottom')
    
    plt.suptitle('Case Align Correlation Analysis Summary', fontsize=16, fontweight='bold')
    plt.show()