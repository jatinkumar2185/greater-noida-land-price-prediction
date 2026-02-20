import pandas as pd
import numpy as np
import os

np.random.seed(42)

SECTOR_CONFIG = {
    "Alpha 1":      {"base": 45000, "metro": (0.5, 2.0), "airport": (22, 28)},
    "Alpha 2":      {"base": 43000, "metro": (0.8, 2.5), "airport": (22, 27)},
    "Beta 1":       {"base": 40000, "metro": (1.0, 3.0), "airport": (20, 26)},
    "Beta 2":       {"base": 38000, "metro": (1.5, 3.5), "airport": (20, 25)},
    "Gamma 1":      {"base": 36000, "metro": (2.0, 4.0), "airport": (18, 24)},
    "Gamma 2":      {"base": 34000, "metro": (2.5, 4.5), "airport": (18, 23)},
    "Delta 1":      {"base": 32000, "metro": (3.0, 5.5), "airport": (16, 22)},
    "Delta 2":      {"base": 30000, "metro": (3.5, 6.0), "airport": (16, 21)},
    "Pari Chowk":   {"base": 50000, "metro": (0.2, 1.5), "airport": (24, 30)},
    "Sector 1":     {"base": 28000, "metro": (4.0, 7.0), "airport": (14, 20)},
    "Sector 2":     {"base": 26000, "metro": (4.5, 7.5), "airport": (12, 18)},
    "Sector 3":     {"base": 24000, "metro": (5.0, 8.0), "airport": (10, 16)},
    "Omicron 1":    {"base": 42000, "metro": (1.0, 3.0), "airport": (20, 26)},
    "Omicron 2":    {"base": 39000, "metro": (1.5, 3.5), "airport": (19, 25)},
    "Zeta 1":       {"base": 35000, "metro": (2.5, 5.0), "airport": (15, 21)},
    "Zeta 2":       {"base": 33000, "metro": (3.0, 5.5), "airport": (14, 20)},
    "Knowledge Park 1": {"base": 37000, "metro": (1.5, 3.5), "airport": (17, 23)},
    "Knowledge Park 2": {"base": 35000, "metro": (2.0, 4.0), "airport": (16, 22)},
    "Techzone IV":  {"base": 29000, "metro": (4.0, 6.5), "airport": (10, 16)},
    "Ecotech 1":    {"base": 27000, "metro": (5.0, 8.5), "airport": (8, 14)},
}

SECTORS = list(SECTOR_CONFIG.keys())
n = 1200
rows = []

for i in range(n):
    sector = np.random.choice(SECTORS)
    cfg    = SECTOR_CONFIG[sector]
    area   = round(float(np.random.choice([
        np.random.randint(50, 200),
        np.random.randint(200, 500),
        np.random.randint(500, 1000),
    ])) + np.random.uniform(-10, 10), 2)
    area = max(50, area)
    road_width    = int(np.random.choice([12, 18, 24, 30, 36, 45, 60, 80], p=[0.1,0.2,0.25,0.2,0.1,0.07,0.05,0.03]))
    metro_dist    = round(np.random.uniform(*cfg["metro"]), 2)
    airport_dist  = round(np.random.uniform(*cfg["airport"]), 2)
    corner_plot   = np.random.choice(["Yes","No"], p=[0.20,0.80])
    facing        = np.random.choice(["North","South","East","West"], p=[0.30,0.20,0.30,0.20])
    nearby_school = np.random.choice(["Yes","No"], p=[0.55,0.45])
    nearby_hosp   = np.random.choice(["Yes","No"], p=[0.45,0.55])
    comm_nearby   = np.random.choice(["Yes","No"], p=[0.40,0.60])

    base_price   = cfg["base"]
    metro_disc   = max(0,(metro_dist-1.0))*3000
    price_psm    = base_price - metro_disc
    price_psm   += (road_width-12)*150
    if corner_plot=="Yes": price_psm*=1.10
    price_psm   *= {"North":1.05,"East":1.03,"South":1.00,"West":0.98}[facing]
    if nearby_school=="Yes": price_psm+=1500
    if nearby_hosp=="Yes":   price_psm+=2000
    if comm_nearby=="Yes":   price_psm+=1800
    price_psm   += max(0,(30-airport_dist))*200
    total_price  = price_psm * area
    noise        = np.random.uniform(0.85,1.15)
    total_price  = round(total_price*noise,-3)
    total_price  = max(500000, total_price)

    rows.append({
        "Sector":sector,"Area_sqm":area,"Road_Width_ft":road_width,
        "Metro_Dist_km":metro_dist,"Airport_Dist_km":airport_dist,
        "Corner_Plot":corner_plot,"Facing":facing,"Nearby_School":nearby_school,
        "Nearby_Hospital":nearby_hosp,"Commercial_Nearby":comm_nearby,
        "Price_INR":int(total_price)
    })

df = pd.DataFrame(rows).sample(frac=1,random_state=42).reset_index(drop=True)
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"dataset.csv"),index=False)
print(f"Dataset generated: {len(df)} rows")
print(f"Price range: {df.Price_INR.min():,} to {df.Price_INR.max():,}")
