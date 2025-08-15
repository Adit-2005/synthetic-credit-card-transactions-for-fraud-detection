import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(n_rows=100000, fraud_rate=0.025):
    np.random.seed(42)
    random.seed(42)

    merchants = ["Amazon", "Walmart", "Target", "Costco", "eBay", "Best Buy", "Apple Store", "Nike", "Adidas", "Starbucks"]
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Miami", "San Francisco", "Seattle", "Boston", "Denver", "Atlanta"]
    states = ["NY", "CA", "IL", "TX", "FL", "CA", "WA", "MA", "CO", "GA"]

    data = []
    start_date = datetime(2023, 1, 1)

    for _ in range(n_rows):
        ts = start_date + timedelta(minutes=random.randint(0, 525600))
        merchant = random.choice(merchants)
        location = f"{random.choice(cities)}, {random.choice(states)}"
        amount = round(random.uniform(1.0, 5000.0), 2)
        card_number = "XXXX-XXXX-XXXX-" + str(random.randint(1000, 9999))
        is_fraud = 1 if random.random() < fraud_rate else 0

        data.append([ts, merchant, location, amount, card_number, is_fraud])

    df = pd.DataFrame(data, columns=["timestamp", "merchant", "location", "amount", "card_number", "is_fraud"])
    return df
