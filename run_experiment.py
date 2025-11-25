# run_experiment.py
import genetic_algorithm_feature_selection as ga_fs
import generate_report
import numpy as np

def run_complete_experiment():
    """Run the complete experiment and generate report"""
    print("Running Genetic Algorithm Feature Selection Experiment...")
    
    # Run the genetic algorithm experiment
    results = ga_fs.main()
    
    # Generate PDF report
    generate_report.create_pdf_report(results)
    
    print("\nExperiment completed successfully!")
    print("Generated files:")
    print("- ga_feature_selection_results.png (Results visualization)")
    print("- TDE3_GA_Feature_Selection_Report.pdf (Complete report)")

if __name__ == "__main__":
    run_complete_experiment()