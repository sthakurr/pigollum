"""
Script to preprocess enantioselectivity (EE) values and convert them to ΔΔG values.

EE (Enantiomeric Excess) Input Format:
    - Values range from -1 to 1 (decimal form)
    - Represents the difference between R and S enantiomers as a fraction
    - Formula: EE = |[R] - [S]| / (|[R] + [S]|)
    - Range: -1 (pure S) to +1 (pure R), with 0 = racemic mixture
    
    In your data:
        - 0.02 means 2% enantiomeric excess (98% racemate, 2% one enantiomer)
        - 0.04 means 4% enantiomeric excess
        - -0.04 means 4% excess of the opposite enantiomer
        
ΔΔG (Delta-Delta-G) Output Format:
    - Thermodynamic measure of selectivity
    - Calculated from: ΔΔG = -RT ln((1 + EE) / (1 - EE))
    - Higher absolute ΔΔG = better selectivity
    - Units: kJ/mol (using R = 8.314 J/(mol·K) and T = 298.15 K)
    
    The relationship is: ΔΔG is proportional to log-odds of one enantiomer over the other
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
import matplotlib.pyplot as plt

def ee_to_ddg(ee_values, temperature=298.15):
    """
    Convert Enantiomeric Excess (EE) values to ΔΔG values.
    
    Parameters:
    -----------
    ee_values : np.ndarray or list
        EE values in decimal form, ranging from -1 to 1
        - Positive values: excess of R enantiomer
        - Negative values: excess of S enantiomer
        - 0: racemic mixture
        
    temperature : float, default=298.15
        Temperature in Kelvin (default is 25°C)
    
    Returns:
    --------
    ddg_values : np.ndarray
        ΔΔG values in kJ/mol
        - Positive ΔΔG: favors R enantiomer
        - Negative ΔΔG: favors S enantiomer
        - Larger absolute value: better selectivity
    
    Notes:
    ------
    - EE values are clipped to [-0.999, 0.999] to avoid log(0)
    - R = 8.314 J/(mol·K) = 0.008314 kJ/(mol·K)
    - Formula: ΔΔG = -RT ln((1 + EE) / (1 - EE))
    """
    
    # Gas constant
    R = 0.008314  # kJ/(mol·K)
    
    # Convert to numpy array if needed
    ee_array = np.asarray(ee_values, dtype=float)
    
    # Clip values to avoid log(0) or negative arguments
    ee_clipped = np.clip(ee_array, -0.999, 0.999)
    
    # Calculate ΔΔG
    # ΔΔG = -RT ln((1 + EE) / (1 - EE))
    ddg_values = -R * temperature * np.log((1 + ee_clipped) / (1 - ee_clipped))
    
    return ddg_values


def preprocess_data(input_csv, output_csv=None, scaling_method='minmax', plot_distributions=True):
    """
    Load EE data from CSV, convert to ΔΔG, and apply scaling/stretching.
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV file containing 'enantioselectivity' column
        
    output_csv : str, optional
        Path to save output CSV. If None, saves as '{input}_with_ddg_scaled.csv'
    
    scaling_method : str, default='minmax'
        Method to stretch/normalize ΔΔG values:
        - 'minmax': Min-Max scaling to [0, 1] range
        - 'standard': Z-score normalization (mean=0, std=1)
        - 'robust': Robust scaling using quantiles (less sensitive to outliers)
        - 'yeo-johnson': Power transformation (handles zero/negative values well)
        - 'box-cox': Power transformation (requires positive values)
        - 'none': No scaling, just ΔΔG conversion
    
    plot_distributions : bool, default=True
        If True, saves a distribution plot showing before/after scaling
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with 'ddg_kJ_mol' and 'ddg_scaled' columns
    """
    
    # Load data
    df = pd.read_csv(input_csv)
    
    if 'enantioselectivity' not in df.columns:
        raise ValueError("CSV must contain 'enantioselectivity' column")
    
    # Convert EE to ΔΔG
    ee_values = df['enantioselectivity'].values
    ddg_values = ee_to_ddg(ee_values)
    
    # Add raw ΔΔG to dataframe
    df['ddg_kJ_mol'] = ddg_values
    
    # Apply scaling
    print(f"\nApplying {scaling_method} scaling...")
    
    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
        ddg_scaled = scaler.fit_transform(ddg_values.reshape(-1, 1)).flatten()
    elif scaling_method == 'standard':
        scaler = StandardScaler()
        ddg_scaled = scaler.fit_transform(ddg_values.reshape(-1, 1)).flatten()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
        ddg_scaled = scaler.fit_transform(ddg_values.reshape(-1, 1)).flatten()
    elif scaling_method == 'yeo-johnson':
        scaler = PowerTransformer(method='yeo-johnson')
        ddg_scaled = scaler.fit_transform(ddg_values.reshape(-1, 1)).flatten()
    elif scaling_method == 'box-cox':
        scaler = PowerTransformer(method='box-cox')
        ddg_scaled = scaler.fit_transform(ddg_values.reshape(-1, 1)).flatten()
    elif scaling_method == 'none':
        ddg_scaled = ddg_values
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    df['ddg_scaled'] = ddg_scaled
    
    # Save
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_with_ddg_scaled.csv')
    
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved preprocessed data to: {output_csv}")
    
    # Print statistics
    print("\n=== Preprocessing Summary ===")
    print(f"Number of samples: {len(df)}")
    print(f"\nRaw ΔΔG (kJ/mol):")
    print(f"  Min: {ddg_values.min():.6f}")
    print(f"  Max: {ddg_values.max():.6f}")
    print(f"  Mean: {ddg_values.mean():.6f}")
    print(f"  Std: {ddg_values.std():.6f}")
    print(f"  Range: {ddg_values.max() - ddg_values.min():.6f}")
    
    print(f"\nScaled ΔΔG ({scaling_method}):")
    print(f"  Min: {ddg_scaled.min():.6f}")
    print(f"  Max: {ddg_scaled.max():.6f}")
    print(f"  Mean: {ddg_scaled.mean():.6f}")
    print(f"  Std: {ddg_scaled.std():.6f}")
    print(f"  Range: {ddg_scaled.max() - ddg_scaled.min():.6f}")
    print(f"  Stretch factor: {(ddg_scaled.max() - ddg_scaled.min()) / (ddg_values.max() - ddg_values.min()):.2f}x")
    
    # Plot distributions if requested
    if plot_distributions:
        plot_path = output_csv.replace('.csv', '_distribution.png')
        _plot_distributions(ddg_values, ddg_scaled, scaling_method, plot_path)
    
    return df


def _plot_distributions(ddg_raw, ddg_scaled, method, output_path):
    """Plot before and after scaling distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw ΔΔG
    axes[0].hist(ddg_raw, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('ΔΔG (kJ/mol)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Before Scaling (Raw ΔΔG)')
    axes[0].grid(alpha=0.3)
    
    # Scaled ΔΔG
    axes[1].hist(ddg_scaled, bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel(f'ΔΔG Scaled ({method})')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'After Scaling ({method})')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved distribution plot to: {output_path}")
    plt.close()


if __name__ == '__main__':
    # Example usage
    input_file = 'data/biocat/enzymes_sequence_yield_ee_processed.csv'
    
    if os.path.exists(input_file):
        # Try different scaling methods
        scaling_methods = ['yeo-johnson']
        
        for method in scaling_methods:
            try:
                output_file = input_file.replace('.csv', f'_ddg_{method}.csv')
                df = preprocess_data(input_file, output_csv=output_file, scaling_method=method, plot_distributions=True)
                print(f"\nFirst few rows with {method} scaling:\n{df[['enantioselectivity', 'ddg_kJ_mol', 'ddg_scaled']].head(5)}\n")
                print("="*70 + "\n")
            except Exception as e:
                print(f"Error with {method} scaling: {e}\n")
    else:
        print(f"Error: {input_file} not found")
        print("Please adjust the input_file path or run from /home/saumya/gollum-2/")
