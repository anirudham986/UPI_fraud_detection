"""
Generate synthetic UPI transaction data
"""
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from config import RANDOM_STATE, FRAUD_RATE

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def generate_synthetic_transactions(n_samples=50000):
    """Generate realistic UPI transaction data with fraud patterns"""

    user_ids = [f'USER_{i:04d}' for i in range(1, 1001)]

    recipients = [f'RECIP_{i:04d}' for i in range(1, 501)]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    data = []
    
    for i in range(n_samples):
        user_id = random.choice(user_ids)
        timestamp = start_date + (end_date - start_date) * random.random()

        is_fraud = random.random() < FRAUD_RATE

        user_avg = 1000 + np.random.exponential(9000)
        
        if is_fraud:
            fraud_type = np.random.choice([
                'micropay_scam', 
                'velocity_attack', 
                'new_recipient', 
                'fake_refund'
            ])
            
            if fraud_type == 'micropay_scam':
                amount = np.random.choice([1, 2, 5, 10])
            elif fraud_type == 'velocity_attack':
                amount = np.random.uniform(5000, 50000)
            elif fraud_type == 'new_recipient':
                amount = user_avg * np.random.uniform(3, 10)
            elif fraud_type == 'fake_refund':
                amount = np.random.uniform(1000, 20000)
                
            recipient = random.choice(recipients)
            
        else:

            amount = max(10, np.random.normal(user_avg, user_avg/3))
            amount = min(amount, 100000)
            
            if random.random() < 0.8:
                recipient = random.choice(recipients[:50])  
            else:
                recipient = random.choice(recipients)  
        
        transaction = {
            'transaction_id': f'TXN_{i:08d}',
            'user_id': user_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'recipient_id': recipient,
            'transaction_type': random.choice(['P2P', 'P2M', 'Recharge', 'Bill Payment']),
            'device_id': f'DEV_{hash(user_id) % 100:03d}',
            'location': random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']),
            'is_fraud': int(is_fraud),
            'fraud_type': fraud_type if is_fraud else 'legitimate'
        }
        
        data.append(transaction)
    
    df = pd.DataFrame(data)
    return df

def add_sequential_patterns(df):
    """Add sequential fraud patterns"""
    users = df['user_id'].unique()[:100] 
    
    for user in users:
        user_mask = df['user_id'] == user
        if user_mask.sum() > 10:

            fraud_indices = df[user_mask].sample(frac=0.1, random_state=RANDOM_STATE).index
            df.loc[fraud_indices, 'is_fraud'] = 1
            df.loc[fraud_indices, 'fraud_type'] = 'velocity_attack'

            user_indices = df[user_mask].index
            if len(user_indices) > 2:
                for i in range(1, len(user_indices) - 1):
                    idx1, idx2 = user_indices[i], user_indices[i + 1]
                    if (df.loc[idx1, 'amount'] <= 10 and 
                        df.loc[idx2, 'amount'] > 10000 and
                        (df.loc[idx2, 'timestamp'] - df.loc[idx1, 'timestamp']).total_seconds() <= 300):
                        df.loc[idx2, 'is_fraud'] = 1
                        df.loc[idx2, 'fraud_type'] = 'micropay_scam'
    
    return df

if __name__ == "__main__":
    print("Generating synthetic UPI transactions...")
    df = generate_synthetic_transactions(50000)
    df = add_sequential_patterns(df)
    
    print(f"Generated {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    
    df.to_csv('data/raw/synthetic_upi_transactions.csv', index=False)
    print("Data saved to data/raw/synthetic_upi_transactions.csv")