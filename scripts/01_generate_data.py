"""
Script 1: Generate synthetic UPI transaction data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import generate_synthetic_transactions, add_sequential_patterns

def main():
    print("="*60)
    print("STEP 1: GENERATE SYNTHETIC UPI TRANSACTION DATA")
    print("="*60)

    print("\nGenerating 50,000 synthetic transactions...")
    df = generate_synthetic_transactions(50000)

    print("Adding sequential fraud patterns...")
    df = add_sequential_patterns(df)
    
    output_path = 'data/raw/synthetic_upi_transactions.csv'
    df.to_csv(output_path, index=False)

    print(f"\n✅ Data generation complete!")
    print(f"Saved to: {output_path}")
    print(f"Total transactions: {len(df):,}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"Fraud types distribution:")
    print(df['fraud_type'].value_counts())
 
    sample_df = df.sample(1000, random_state=42)
    sample_path = 'data/processed/demo_sample.csv'
    sample_df.to_csv(sample_path, index=False)
    print(f"\nCreated demo sample: {sample_path} (1,000 transactions)")

if __name__ == "__main__":
    main()