import pandas as pd
from sqlalchemy import create_engine


db_connection_str = "postgresql://deepakachyutha@localhost:5432/uscrashdata"
engine = create_engine(db_connection_str)


print("Fetching PRE-COVID Control Data (March 2019)...")

# We select ONLY March 2019 to compare with March 2022/23
query_old = """
SELECT 
    "Severity", "Start_Lat", "Start_Lng", 
    "TemperatureF", "Visibilitymi", "Weather_Condition",
    EXTRACT(HOUR FROM "Start_Time"::timestamp) as "Hour",
    EXTRACT(MONTH FROM "Start_Time"::timestamp) as "Month",
    EXTRACT(ISODOW FROM "Start_Time"::timestamp) as "Weekday"
FROM processed_accidents
WHERE EXTRACT(YEAR FROM "Start_Time"::timestamp) = 2019
  AND EXTRACT(MONTH FROM "Start_Time"::timestamp) = 3
ORDER BY RANDOM()
LIMIT 100000;
"""

df_old = pd.read_sql(query_old, engine)
print(f"Loaded {len(df_old)} rows of Control Data (2019).")

# Save raw for analysis (No need to scale yet, we compare distributions)
df_old.to_csv("control_data_2019.csv", index=False)
print("'control_data_2019.csv' created.")