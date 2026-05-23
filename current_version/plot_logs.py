import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_loc_complexity():
    df = pd.read_csv('results LOC count.tsv', sep='\t')
    
    # Filter out crashes if needed, or map statuses to colors
    color_map = {'PASS': 'green', 'pass': 'green', 'FAIL': 'red', 'fail': 'red', 'crash': 'black'}
    df['color'] = df['status'].map(color_map).fillna('gray')
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    for status, group in df.groupby('status'):
        c = color_map.get(status.lower(), 'gray')
        m = 'x' if status.lower() == 'crash' else 'o'
        alpha = 0.6 if status.lower() == 'fail' else 1.0
        plt.scatter(group['generation'], group['lines'], c=c, marker=m, label=status.upper(), alpha=alpha, edgecolors='k' if m == 'o' else None)
    
    # Track the best (lowest) lines that PASS
    passed = df[df['status'].str.lower() == 'pass']
    if not passed.empty:
        best_lines = passed.groupby('generation')['lines'].min().cummin()
        plt.plot(best_lines.index, best_lines.values, 'g--', label='Best Passing LOC')
        
    plt.title('Line Count Complexity over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Lines of Code (LOC)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('loc_complexity.png', dpi=300)
    print("Saved loc_complexity.png")

def plot_new_system():
    df = pd.read_csv('results.tsv', sep='\t')
    
    # Replace inf with NaN for combined_score and drop or handle them
    df['combined_score'] = pd.to_numeric(df['combined_score'], errors='coerce')
    
    color_map = {'PASS': 'green', 'pass': 'green', 'FAIL': 'red', 'fail': 'red', 'crash': 'black'}
    
    plt.figure(figsize=(10, 6))
    
    for status, group in df.groupby('status'):
        # filter out nan
        group = group.dropna(subset=['combined_score'])
        if group.empty:
            continue
        c = color_map.get(status.lower(), 'gray')
        m = 'x' if status.lower() == 'crash' else 'o'
        alpha = 0.6 if status.lower() == 'fail' else 1.0
        plt.scatter(group['generation'], group['combined_score'], c=c, marker=m, label=status.upper(), alpha=alpha, edgecolors='k' if m == 'o' else None)
    
    # Track the best (lowest) combined_score that PASS
    passed = df[df['status'].str.lower() == 'pass']
    passed = passed.dropna(subset=['combined_score'])
    if not passed.empty:
        best_scores = passed.groupby('generation')['combined_score'].min().cummin()
        plt.plot(best_scores.index, best_scores.values, 'g--', label='Best Passing Score')
        
    plt.title('New System Complexity (Combined Score) over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Combined Complexity Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('new_system_complexity.png', dpi=300)
    print("Saved new_system_complexity.png")

if __name__ == '__main__':
    plot_loc_complexity()
    plot_new_system()
