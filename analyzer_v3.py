import pandas as pd
import numpy as np

class PatientAnalyzer:
    def __init__(self, csv_path=None, df=None):
        # Imports the CSV as a pandas DataFrame or accepts a DataFrame directly
        if df is not None:
            self.df = df
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Either csv_path or df must be provided")

    def get_strategic_profile(self):
        """Filters for cancer cells and calculates their strength based on raw data."""
        
        cancer_subset = self.df[self.df['Disease_Status'] == 'Tumor']
    
        if cancer_subset.empty:
            return {"avg_growth": 14.0, "max_res_a": 15.0, "tox_increment": 1.0}

        # 1. Max Efficacy dari Gene_D (Therapy target)
        base_eff = cancer_subset['Gene_D_Therapy'].mean() / 20.0
        max_eff_a = np.clip(base_eff, 0.7, 0.95)
        
        avg_immune = cancer_subset['Gene_B_Immune'].mean()
        dynamic_res_b = max(1.0, 15.0 - avg_immune)
        
        stromal_score = cancer_subset['Gene_C_Stromal'].mean()
        # Logika: Jika Stromal rendah (misal 5), kenaikan toksisitas tinggi (misal 1.5)
        # Jika Stromal tinggi (misal 15), kenaikan toksisitas rendah (misal 0.5)
        dynamic_tox_inc = max(0.4, 2.0 - (stromal_score / 10.0))

        return {
            "normalized_init_tumor_size": 1,
            "initial_tumor_size": len(cancer_subset),
            "avg_growth": cancer_subset['Gene_A_Oncogene'].mean(),
            "max_res_a": cancer_subset['Gene_D_Therapy'].max(),
            "starting_res_a": cancer_subset['Gene_D_Therapy'].mean(),
            "starting_res_b": float(dynamic_res_b),
            "max_efficacy_a": max_eff_a,
            "tox_increment": float(dynamic_tox_inc),
            "trap_sensitivity": avg_immune / 20.0 # Kita asumsikan jika skor imun awal sudah ada, 
            # maka Obat A akan lebih mudah mengekspos sel tersebut lebih jauh.
        }

    def get_patient_profile(self, data=None):
        """Alias for get_strategic_profile for compatibility."""
        if data is not None:
            self.df = data
        return self.get_strategic_profile()
    
    def get_cell_resistance_data(self):
        """Returns individual cell resistance levels from tumor cells."""
        # Filter for cancer cells only
        cancer_subset = self.df[self.df['Disease_Status'] == 'Tumor']
        
        if cancer_subset.empty:
            return []
        
        # Return resistance levels (Gene_D_Therapy) for each tumor cell
        # Normalize to 0-15 scale for consistency with max_res_a
        resistance_values = cancer_subset['Gene_D_Therapy'].values.tolist()
        return resistance_values

# Example of how to run it:
if __name__ == "__main__":
    # Point to your synthetic CSV file
    analyzer = PatientAnalyzer("data/Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv")
    
    # Get the calculated values
    profile = analyzer.get_strategic_profile()
    
    # Print the results
    print("Strategic Profile:")
    for key, value in profile.items():
        print(f"{key}: {value:.4f}")