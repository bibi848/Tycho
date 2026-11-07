#%%
# Importing modules and data
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

DATA_FILE = Path.cwd() / "Data" / "IPA_properties.csv"

df = pd.read_csv(DATA_FILE, skipinitialspace=True)
df.head()


#%%
# Plot Cp against temperature, one line per pressure
fig, ax = plt.subplots()
for p, group in df.groupby("Pressure_MPA"):
    ax.plot(group["Temperature_K"], group["Cp_kJkgK"], marker="o", label=f"{p:g} MPa")
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("Cp [kJ/(kg K)]")
ax.set_title("IPA: Cp vs Temperature")
ax.legend()
ax.grid(True)
plt.show()
