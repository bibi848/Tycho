#%%
# Importing modules and data
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data_IPA"
IMG_DIR = ROOT / "images"
IMG_DIR.mkdir(exist_ok=True)

props = ["Cp_kJkgK", "Density_kgm3", "Viscosity_mewPas", "ThermalConductivity_mWmK"]
labels = ["Cp [kJ/(kg K)]", "Density [kg/m3]", "Viscosity [uPa s]", "Thermal conductivity [mW/(m K)]"]

# Load given data (RPA)
rpa = pd.read_csv(DATA_DIR / "IPA_properties.csv", skipinitialspace=True)

# Chemcast data, one file per pressure, converted to the RPA units
chemcast_files = {0.101325: "1atm", 1.0: "1MPa", 6.0: "6MPa", 15.0: "15MPa", 30.0: "30MPa"}
frames = []
for p, tag in chemcast_files.items():
    d = pd.read_csv(DATA_DIR / f"reference_Chemcast_{tag}.csv", comment="#",
                    sep=r"\s*,\s*", engine="python", na_values="N/A")
    d["Pressure_MPA"] = p
    frames.append(d)
cc = pd.concat(frames, ignore_index=True)
cc["Phase"] = cc["Phase"].str.strip()
cc["Temperature_K"] = cc["temperature_K"]
cc["Cp_kJkgK"] = cc["Cp_JkgK"] / 1000
cc["Density_kgm3"] = cc["density_kgm3"]
cc["Viscosity_mewPas"] = cc["dynamicViscosity_Pas"] * 1e6
cc["ThermalConductivity_mWmK"] = cc["thermalConductivity_WmK"] * 1000

# Celsius data, saturated liquid 273-408 K
celsius = pd.read_csv(DATA_DIR / "reference_Celsius.csv", comment="#", skipinitialspace=True)
celsius["Temperature_K"] = celsius["temperature_C"] + 273.15
celsius["Cp_kJkgK"] = celsius["Cp_JkgK"] / 1000
celsius["Density_kgm3"] = celsius["density_kgm3"]
celsius["Viscosity_mewPas"] = celsius["dynamicViscosity_Pas"] * 1e6
celsius["ThermalConductivity_mWmK"] = celsius["thermalConductivity_WmK"] * 1000

# VDI Heat Atlas data, saturated liquid 356-508 K
vdi = pd.read_csv(DATA_DIR / "reference_VDI_IPA_saturation.csv", comment="#")
vdi["Cp_kJkgK"] = vdi["Cp_l_kJkgK"]
vdi["Density_kgm3"] = vdi["Density_l_kgm3"]
vdi["Viscosity_mewPas"] = vdi["Viscosity_l_uPas"]
vdi["ThermalConductivity_mWmK"] = vdi["ThermalCond_l_mWmK"]

pressures = list(chemcast_files)
colours = dict(zip(pressures, plt.rcParams["axes.prop_cycle"].by_key()["color"]))


#%%
# Plot Cp against temperature (RPA file only)
fig, ax = plt.subplots()
for p, group in rpa.groupby("Pressure_MPA"):
    ax.plot(group["Temperature_K"], group["Cp_kJkgK"], marker="o", color=colours[p], label=f"{p:g} MPa")
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("Cp [kJ/(kg K)]")
ax.set_title("IPA: Cp vs Temperature (RPA file)")
ax.legend()
ax.grid(True)
fig.savefig(IMG_DIR / "rpa_cp_vs_T.png", dpi=150)
plt.show()


#%%
# Sanity check on Chemcast: liquid at 1 atm vs Celsius, % difference (only where both have data)
liq_1atm = cc[(cc["Pressure_MPA"] == 0.101325) & (cc["Phase"] == "l")]
check = pd.DataFrame({"Temperature_K": liq_1atm["Temperature_K"]})
for prop in props:
    c = celsius.dropna(subset=[prop])
    cel_at_T = np.interp(liq_1atm["Temperature_K"], c["Temperature_K"], c[prop], left=np.nan, right=np.nan)
    check[prop] = 100 * (liq_1atm[prop] / cel_at_T - 1)
print("Chemcast vs Celsius, liquid at 1 atm, % difference")
print(check.dropna().round(1).to_string(index=False))


#%%
# Compare RPA against Chemcast at the same pressure and in the same phase
# Chemcast is interpolated in temperature; RPA points in the gap between Chemcast's last liquid and first gas point are skipped
compare = rpa[["Pressure_MPA", "Temperature_K"] + props].copy()
for p in pressures:
    for phase in ["l", "g"]:
        ref = cc[(cc["Pressure_MPA"] == p) & (cc["Phase"] == phase)]
        rows = (compare["Pressure_MPA"] == p) & compare["Temperature_K"].between(ref["Temperature_K"].min(), ref["Temperature_K"].max())
        compare.loc[rows, "Phase"] = phase
        for prop in props:
            r = ref.dropna(subset=[prop])
            compare.loc[rows, prop + "_chemcast"] = np.interp(compare.loc[rows, "Temperature_K"], r["Temperature_K"], r[prop])

# Chemcast's gas phase looks like an ideal gas, so it is only trusted at 1 atm
trusted = (compare["Phase"] == "l") | (compare["Pressure_MPA"] == 0.101325)

for prop in props:
    compare[prop + "_err_%"] = np.where(trusted, 100 * (compare[prop] / compare[prop + "_chemcast"] - 1), np.nan)
    table = compare.pivot(index="Temperature_K", columns="Pressure_MPA", values=prop + "_err_%")
    print(f"\n{prop}: RPA vs Chemcast, % difference")
    print(table.round(1).to_string())


#%%
# Plot: RPA points vs Chemcast curves at every pressure, with Celsius and VDI saturated liquid as a cross-check
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, prop, label in zip(axes.flat, props, labels):
    for p in pressures:
        for phase, ls in [("l", "-"), ("g", "--")]:
            ref = cc[(cc["Pressure_MPA"] == p) & (cc["Phase"] == phase)]
            ax.plot(ref["Temperature_K"], ref[prop], ls, color=colours[p], lw=1)
        r = rpa[rpa["Pressure_MPA"] == p]
        ax.plot(r["Temperature_K"], r[prop], "o", color=colours[p], mfc="none", label=f"RPA {p:g} MPa")
    ax.plot(celsius["Temperature_K"], celsius[prop], "kx", label="Celsius (sat. liquid)")
    ax.plot(vdi["Temperature_K"], vdi[prop], "ks", ms=4, label="VDI (sat. liquid)")
    ax.set_xlim(240, 610)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(label)
    ax.grid(True)
axes[0, 1].set_yscale("log")
axes[1, 0].set_yscale("log")
axes[0, 0].legend(fontsize=8)
fig.suptitle("IPA: RPA file (circles) vs Chemcast (lines: solid = liquid, dashed = gas)")
fig.tight_layout()
fig.savefig(IMG_DIR / "rpa_vs_chemcast.png", dpi=150)
plt.show()


#%%
# Plot: % difference of RPA from Chemcast
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax, prop, label in zip(axes.flat, props, labels):
    for p, g in compare.groupby("Pressure_MPA"):
        ax.plot(g["Temperature_K"], g[prop + "_err_%"], "o-", color=colours[p], label=f"{p:g} MPa")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(label.split(" [")[0] + " error [%]")
    ax.grid(True)
for ax in axes[1]:
    ax.set_xlabel("Temperature [K]")
axes[0, 0].legend()
fig.suptitle("IPA: RPA file vs Chemcast, % difference")
fig.tight_layout()
fig.savefig(IMG_DIR / "rpa_error_vs_chemcast.png", dpi=150)
plt.show()
