import pandas as pd
import numpy as np


def parse_ee_string(ee_str):
    """
    Convert enantioselectivity string to signed value.
    
    Args:
        ee_str: String in format "R:S%" (e.g., "42:58%") or special value "No e.e."
        
    Returns:
        float: Signed ee value using formula (R-S)/(R+S), or 0.0 for racemic
    """
    if pd.isna(ee_str):
        return np.nan
    
    ee_str = str(ee_str).strip()
    
    # Handle "No e.e." (racemic mixture)
    if ee_str.lower() in ["no e.e.", "no ee", "racemic"]:
        return 0.0
    
    try:
        # Parse "R:S%" format
        # Remove '%' if present
        ee_str = ee_str.rstrip(" e.r.").strip()
        
        # Split by ':' to get R and S percentages
        parts = ee_str.split(":")
        if len(parts) != 2:
            return np.nan
        
        r_pct = float(parts[0].strip())
        s_pct = float(parts[1].strip())
        
        # Apply formula: (R-S)/(R+S)
        total = r_pct + s_pct
        if total == 0:
            return np.nan
        
        signed_ee = (r_pct - s_pct) / total
        return signed_ee
    
    except (ValueError, AttributeError):
        return np.nan


def load_biocat_data(data_path):
    """
    Load biocat data and convert enantioselectivity values to signed format.
    
    Args:
        data_path: Path to CSV file with 'enantioselectivity' column
        
    Returns:
        pandas.DataFrame: Data with 'enantioselectivity' column converted to signed values
    """
    data = pd.read_csv(data_path)
    
    # Convert ee strings to signed values
    data["enantioselectivity"] = data["enantioselectivity"].apply(parse_ee_string)
    
    # save the processed data to a new CSV file
    processed_data_path = data_path.replace(".csv", "_processed.csv")
    data.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    data_path = "enzymes_sequence_yield_ee.csv"
    load_biocat_data(data_path)
