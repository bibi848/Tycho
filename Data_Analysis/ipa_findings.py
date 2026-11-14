#%%
# Cross-check of IPA_properties.csv (RPA input) against all reference sources in Data_IPA.
# Produces the summary figures used in the Obsidian note "Checking IPA Setup".
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data_IPA"
IMG_DIR = ROOT / "images"
IMG_DIR.mkdir(exist_ok=True)

MW = 60.096  # kg/kmol
R = 8.314462618  # J/(mol K)

# RPA file
rpa = pd.read_csv(DATA_DIR / "IPA_properties.csv", skipinitialspace=True)

# Chemcast (same loading as data_check.py)
chemcast_files = {0.101325: "1atm", 1.0: "1MPa", 6.0: "6MPa", 15.0: "15MPa", 30.0: "30MPa"}
frames = []
for p, tag in chemcast_files.items():
    d = pd.read_csv(DATA_DIR / f"reference_Chemcast_{tag}.csv", comment="#",
                    sep=r"\s*,\s*", engine="python", na_values="N/A")
    d["Pressure_MPA"] = p
    frames.append(d)
cc = pd.concat(frames, ignore_index=True)
cc["Phase"] = cc["Phase"].str.strip()
cc["T"] = cc["temperature_K"]
cc["Cp"] = cc["Cp_JkgK"] / 1000
cc["rho"] = cc["density_kgm3"]
cc["mu"] = cc["dynamicViscosity_Pas"] * 1e6
cc["k"] = cc["thermalConductivity_WmK"] * 1000

# Celsius (saturated liquid)
cel = pd.read_csv(DATA_DIR / "reference_Celsius.csv", comment="#", skipinitialspace=True)
cel["T"] = cel["temperature_C"] + 273.15
cel["Cp"] = cel["Cp_JkgK"] / 1000
cel["rho"] = cel["density_kgm3"]
cel["mu"] = cel["dynamicViscosity_Pas"] * 1e6
cel["k"] = cel["thermalConductivity_WmK"] * 1000

# VDI Heat Atlas (saturated liquid and vapour)
vdi = pd.read_csv(DATA_DIR / "reference_VDI_IPA_saturation.csv", comment="#")

# NIST ideal-gas Cp
nist = pd.read_csv(DATA_DIR / "reference_NIST_IPA_ideal_gas_Cp.csv", comment="#")

# DIPPR correlations (Perry's 8th ed.)
dippr = pd.read_csv(DATA_DIR / "reference_DIPPR_coefficients.csv", comment="#")
dippr = dippr[dippr["Fluid"] == "IPA"].set_index("Property")


def dippr_eval(prop, T):
    """Evaluate a DIPPR correlation; returns NaN outside [Tmin, Tmax]."""
    r = dippr.loc[prop]
    C1, C2, C3, C4, C5 = r[["C1", "C2", "C3", "C4", "C5"]]
    T = np.asarray(T, dtype=float)
    eq = int(r["DIPPR_eq"])
    if eq == 100:
        y = C1 + C2 * T + C3 * T**2 + C4 * T**3 + C5 * T**4
    elif eq == 101:
        y = np.exp(C1 + C2 / T + C3 * np.log(T) + C4 * T**C5)
    elif eq == 102:
        y = C1 * T**C2 / (1 + C3 / T + C4 / T**2)
    elif eq == 105:
        y = C1 / C2 ** (1 + (1 - T / C3) ** C4)
    elif eq == 106:
        Tr = T / r["Tc"]
        y = C1 * (1 - Tr) ** (C2 + C3 * Tr + C4 * Tr**2)
    return np.where((T >= r["Tmin"]) & (T <= r["Tmax"]), y, np.nan)


def dippr_si(prop, T):
    """DIPPR values in the RPA units."""
    scale = {"Cp_liquid": 1 / MW / 1000, "rho_liquid": MW, "mu_liquid": 1e6,
             "mu_vapour": 1e6, "k_liquid": 1e3, "k_vapour": 1e3, "Hvap": 1 / MW / 1000 * 1000}
    return dippr_eval(prop, T) * scale[prop]


rpa_liq_1atm = rpa[(rpa["Pressure_MPA"] == 0.101325) & (rpa["Temperature_K"] < 355)]
rpa_gas_1atm = rpa[(rpa["Pressure_MPA"] == 0.101325) & (rpa["Temperature_K"] > 356)]
# Liquid Cp/mu/k are identical at every pressure in the RPA file, so the 30 MPa block gives the full liquid range
rpa_liq_all = rpa[(rpa["Pressure_MPA"] == 30.0) & (rpa["Temperature_K"] <= 493)]

liq_cc = {p: cc[(cc["Pressure_MPA"] == p) & (cc["Phase"] == "l")] for p in chemcast_files}
gas_cc_1atm = cc[(cc["Pressure_MPA"] == 0.101325) & (cc["Phase"] == "g")]

colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]


#%%
# Structural checks on the RPA file
print("Liquid Cp, viscosity, conductivity identical at every pressure where liquid:")
liq = rpa[rpa["Temperature_K"] <= 413]
for col in ["Cp_kJkgK", "Viscosity_mewPas", "ThermalConductivity_mWmK"]:
    spread = liq.groupby("Temperature_K")[col].nunique().max()
    print(f"  {col}: max distinct values per temperature = {spread}")
print("Liquid Cp step per 20 K:", np.unique(np.round(np.diff(rpa_liq_all["Cp_kJkgK"]), 4)))
print("Gas Cp / mu / k step per 20 K at 1 atm:",
      np.unique(np.round(np.diff(rpa_gas_1atm["Cp_kJkgK"]), 4)),
      np.unique(np.round(np.diff(rpa_gas_1atm["Viscosity_mewPas"]), 4)),
      np.unique(np.round(np.diff(rpa_gas_1atm["ThermalConductivity_mWmK"]), 4)))

# Saturation data
print("\nSaturation temperature and heat of vaporisation")
for p, Tsat, Hv in rpa.dropna(subset=["TempVapor_Tsp_K"])[["Pressure_MPA", "TempVapor_Tsp_K", "HeatVapor_QsTs_kJkg"]].values:
    Ts = np.arange(300, 508.3, 0.01)
    ok = Ts[dippr_eval("Psat", Ts) >= p * 1e6]
    if len(ok):
        print(f"  {p:g} MPa: RPA Tsat {Tsat:.2f} K, Hvap {Hv:.1f} kJ/kg | DIPPR Tsat {ok[0]:.2f} K, Hvap {dippr_si('Hvap', ok[0]):.1f} kJ/kg"
              f" | VDI Tsat {np.interp(p, vdi['Pressure_MPa'], vdi['Temperature_K']):.2f} K, Hvap {np.interp(p, vdi['Pressure_MPa'], vdi['Hvap_kJkg']):.1f} kJ/kg")
    else:
        print(f"  {p:g} MPa: RPA Tsat {Tsat:.2f} K, Hvap {Hv:.1f} -> above critical pressure (Pc ~ 4.76 MPa), treated as supercritical")


#%%
# Liquid: RPA vs each source, % difference (low pressure / saturated liquid)
T = rpa_liq_all["Temperature_K"].to_numpy()
vdi_l = {"Cp": "Cp_l_kJkgK", "rho": "Density_l_kgm3", "mu": "Viscosity_l_uPas", "k": "ThermalCond_l_mWmK"}
rpa_cols = {"Cp": "Cp_kJkgK", "rho": "Density_kgm3", "mu": "Viscosity_mewPas", "k": "ThermalConductivity_mWmK"}
dippr_names = {"Cp": "Cp_liquid", "rho": "rho_liquid", "mu": "mu_liquid", "k": "k_liquid"}


def interp_nan(x, xp, fp):
    m = ~np.isnan(fp)
    return np.interp(x, xp[m], fp[m], left=np.nan, right=np.nan)


rows = []
for prop in ["Cp", "mu", "k"]:
    r = rpa_liq_all[rpa_cols[prop]].to_numpy()
    refs = {
        "Celsius": interp_nan(T, cel["T"].to_numpy(), cel[prop].to_numpy()),
        "VDI": interp_nan(T, vdi["Temperature_K"].to_numpy(), vdi[vdi_l[prop]].to_numpy()),
        "DIPPR": dippr_si(dippr_names[prop], T),
        "Chemcast 1 atm/1 MPa": np.where(T < 355, interp_nan(T, liq_cc[0.101325]["T"].to_numpy(), liq_cc[0.101325][prop].to_numpy()),
                                          interp_nan(T, liq_cc[1.0]["T"].to_numpy(), liq_cc[1.0][prop].to_numpy())),
    }
    for name, ref in refs.items():
        for t, a, b in zip(T, r, ref):
            if not np.isnan(b):
                rows.append(dict(prop=prop, source=name, T=t, err=100 * (a / b - 1)))
liq_err = pd.DataFrame(rows)
print("\nLiquid (low pressure): RPA % difference from each source")
print(liq_err.pivot_table(index="T", columns=["prop", "source"], values="err").round(1).to_string())


#%%
# Vapour at 1 atm: RPA vs NIST / DIPPR / VDI / Chemcast
Tg = rpa_gas_1atm["Temperature_K"].to_numpy()
rho_ideal = 101325 * MW / 1000 / (R * Tg)
print("\nVapour at 1 atm, RPA % difference")
gas = pd.DataFrame({
    "T": Tg,
    "Cp vs NIST ideal gas": 100 * (rpa_gas_1atm["Cp_kJkgK"].to_numpy() / np.interp(Tg, nist["Temperature_K"], nist["Cp_kJkgK"]) - 1),
    "mu vs DIPPR": 100 * (rpa_gas_1atm["Viscosity_mewPas"].to_numpy() / dippr_si("mu_vapour", Tg) - 1),
    "mu vs Chemcast": 100 * (rpa_gas_1atm["Viscosity_mewPas"].to_numpy() / interp_nan(Tg, gas_cc_1atm["T"].to_numpy(), gas_cc_1atm["mu"].to_numpy()) - 1),
    "k vs DIPPR": 100 * (rpa_gas_1atm["ThermalConductivity_mWmK"].to_numpy() / dippr_si("k_vapour", Tg) - 1),
    "k vs Chemcast": 100 * (rpa_gas_1atm["ThermalConductivity_mWmK"].to_numpy() / interp_nan(Tg, gas_cc_1atm["T"].to_numpy(), gas_cc_1atm["k"].to_numpy()) - 1),
    "rho vs ideal gas": 100 * (rpa_gas_1atm["Density_kgm3"].to_numpy() / rho_ideal - 1),
})
print(gas.round(1).to_string(index=False))
print("VDI sat. vapour at 373 K: mu", vdi.loc[1, "Viscosity_g_uPas"], "k", vdi.loc[1, "ThermalCond_g_mWmK"],
      "| RPA 373 K 1 atm: mu", rpa_gas_1atm["Viscosity_mewPas"].iloc[0], "k", rpa_gas_1atm["ThermalConductivity_mWmK"].iloc[0])


#%%
# Compressed liquid density must be >= saturated liquid density at the same temperature
print("\nRPA liquid density at 6 MPa vs saturated liquid density")
r6 = rpa[(rpa["Pressure_MPA"] == 6.0) & (rpa["Temperature_K"].between(413, 493))]
T6 = r6["Temperature_K"].to_numpy()
print(pd.DataFrame({
    "T": T6,
    "RPA 6 MPa": r6["Density_kgm3"].to_numpy(),
    "DIPPR sat": dippr_si("rho_liquid", T6).round(1),
    "VDI sat": interp_nan(T6, vdi["Temperature_K"].to_numpy(), vdi["Density_l_kgm3"].to_numpy()).round(1),
    "Chemcast 6 MPa": interp_nan(T6, liq_cc[6.0]["T"].to_numpy(), liq_cc[6.0]["rho"].to_numpy()).round(1),
}).to_string(index=False))


#%%
# Figure 1: liquid Cp
Tfine = np.linspace(250, 500, 300)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(rpa_liq_all["Temperature_K"], rpa_liq_all["Cp_kJkgK"], "ko-", mfc="none", label="RPA file (all pressures)")
ax.plot(cel["T"], cel["Cp"], "x", color=colours[0], label="Celsius (sat. liquid)")
ax.plot(vdi["Temperature_K"], vdi["Cp_l_kJkgK"], "s", color=colours[1], label="VDI Heat Atlas (sat. liquid)")
ax.plot(Tfine, dippr_si("Cp_liquid", Tfine), "-", color=colours[2], label="DIPPR (valid to 355 K)")
ax.plot(liq_cc[1.0]["T"], liq_cc[1.0]["Cp"], "--", color=colours[3], label="Chemcast 1 MPa")
ax.plot(liq_cc[6.0]["T"], liq_cc[6.0]["Cp"], ":", color=colours[4], label="Chemcast 6 MPa")
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("Liquid Cp [kJ/(kg K)]")
ax.set_title("IPA liquid Cp: RPA file vs references")
ax.grid(True)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(IMG_DIR / "ipa_liquid_cp.png", dpi=150)
plt.show()


#%%
# Figure 2: vapour at 1 atm (Cp, viscosity, conductivity)
Tg_fine = np.linspace(355, 600, 200)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
specs = [
    ("Cp_kJkgK", "Cp", "Cp [kJ/(kg K)]", None, None),
    ("Viscosity_mewPas", "mu", "Viscosity [uPa s]", "mu_vapour", "Viscosity_g_uPas"),
    ("ThermalConductivity_mWmK", "k", "Thermal conductivity [mW/(m K)]", "k_vapour", "ThermalCond_g_mWmK"),
]
for ax, (col, prop, label, dname, vcol) in zip(axes, specs):
    ax.plot(rpa_gas_1atm["Temperature_K"], rpa_gas_1atm[col], "ko-", mfc="none", label="RPA file 1 atm")
    ax.plot(gas_cc_1atm["T"], gas_cc_1atm[prop], "--", color=colours[3], label="Chemcast 1 atm")
    if dname:
        ax.plot(Tg_fine, dippr_si(dname, Tg_fine), "-", color=colours[2], label="DIPPR")
        ax.plot(vdi["Temperature_K"].iloc[:1], vdi[vcol].iloc[:1], "s", color=colours[1], label="VDI sat. vapour 1 atm")
    else:
        ax.plot(nist["Temperature_K"], nist["Cp_kJkgK"], "^", color=colours[0], label="NIST ideal gas")
    ax.set_xlim(350, 600)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(label)
    ax.grid(True)
    ax.legend(fontsize=8)
fig.suptitle("IPA vapour at 1 atm: RPA file vs references")
fig.tight_layout()
fig.savefig(IMG_DIR / "ipa_vapour_1atm.png", dpi=150)
plt.show()


#%%
# Figure 3: summary of % error in the liquid (consensus of low-pressure sources)
fig, ax = plt.subplots(figsize=(7, 4.5))
names = {"Cp": "Cp", "mu": "Viscosity", "k": "Thermal conductivity"}
for i, prop in enumerate(["Cp", "mu", "k"]):
    g = liq_err[liq_err["prop"] == prop].groupby("T")["err"].agg(["median", "min", "max", "count"])
    g = g[g["count"] >= 2]  # only temperatures with at least two sources
    ax.plot(g.index, g["median"], "o-", color=colours[i], label=f"{names[prop]} (median of sources)")
    ax.fill_between(g.index, g["min"], g["max"], color=colours[i], alpha=0.2)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("RPA error [%]")
ax.set_title("IPA liquid, low pressure: RPA error vs Celsius / VDI / DIPPR / Chemcast\n(band = spread between sources, only T with >= 2 sources)", fontsize=10)
ax.grid(True)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(IMG_DIR / "ipa_liquid_error_summary.png", dpi=150)
plt.show()


#%%
# Figure 4: liquid density at high pressure near the critical temperature
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, p in enumerate([1.0, 6.0, 30.0]):
    r = rpa[(rpa["Pressure_MPA"] == p) & (rpa["Density_kgm3"] > 300)]
    ax.plot(r["Temperature_K"], r["Density_kgm3"], "o", mfc="none", color=colours[i], label=f"RPA {p:g} MPa")
    ax.plot(liq_cc[p]["T"], liq_cc[p]["rho"], "-", color=colours[i], lw=1, label=f"Chemcast {p:g} MPa")
ax.plot(vdi["Temperature_K"], vdi["Density_l_kgm3"], "ks", ms=4, label="VDI sat. liquid")
ax.plot(Tfine, dippr_si("rho_liquid", Tfine), "k--", lw=1, label="DIPPR sat. liquid")
ax.set_xlim(300, 510)
ax.set_ylim(400, 820)
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("Liquid density [kg/m3]")
ax.set_title("IPA liquid density: compressed liquid must lie above the saturated curve")
ax.grid(True)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(IMG_DIR / "ipa_liquid_density.png", dpi=150)
plt.show()
