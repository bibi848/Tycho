#%%
# Build corrected IPA property files for RPA.
#   In : Data_IPA/IPA_properties.csv, Data_IPA/usr_properties.txt (original) + reference data
#   Out: New_IPA/IPA_properties.csv, New_IPA/usr_properties.txt, New_IPA/changes.csv
#
# Changes (agreed after the check in ipa_findings.py):
#   1. Liquid Cp      -> DIPPR (T < 355 K), VDI Heat Atlas sat. liquid (T >= 355 K),
#                        VDI linearly extrapolated to 493 K (VDI Cp stops at 478 K)
#   2. Vapour mu, k   -> DIPPR vapour correlations (Perry's 8th ed.)
#   3. Liquid density -> Chemcast at the same pressure, wherever the original value is below
#                        the DIPPR saturated-liquid density (compressed liquid cannot be less dense)
# Everything else is copied unchanged, including the supercritical rows (T > Tc = 508.3 K at 6/15/30 MPa).
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path.cwd() / "Data_IPA"
OUT_DIR = Path.cwd() / "New_IPA"
OUT_DIR.mkdir(exist_ok=True)

MW = 60.096
TC = 508.3  # K, critical temperature of IPA (DIPPR / IUPAC)
COLS = ["Pressure_MPA", "Temperature_K", "Cp_kJkgK", "Density_kgm3", "Viscosity_mewPas",
        "ThermalConductivity_mWmK", "TempVapor_Tsp_K", "HeatVapor_QsTs_kJkg"]

# ---------------------------------------------------------------- reference data
dippr = pd.read_csv(DATA_DIR / "reference_DIPPR_coefficients.csv", comment="#")
dippr = dippr[dippr["Fluid"] == "IPA"].set_index("Property")


def dippr_eval(prop, T):
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
    if np.any((T < r["Tmin"]) | (T > r["Tmax"])):
        raise ValueError(f"{prop}: T outside DIPPR validity range {r['Tmin']}-{r['Tmax']} K")
    return y


vdi = pd.read_csv(DATA_DIR / "reference_VDI_IPA_saturation.csv", comment="#")
vdi_cp = vdi.dropna(subset=["Cp_l_kJkgK"])

chemcast_files = {0.101325: "1atm", 1.0: "1MPa", 6.0: "6MPa", 15.0: "15MPa", 30.0: "30MPa"}
cc_liq = {}
for p, tag in chemcast_files.items():
    d = pd.read_csv(DATA_DIR / f"reference_Chemcast_{tag}.csv", comment="#",
                    sep=r"\s*,\s*", engine="python", na_values="N/A")
    d = d[d["Phase"].str.strip() == "l"].dropna(subset=["density_kgm3"])
    cc_liq[p] = d


def liquid_cp(T):
    """Liquid Cp [kJ/(kg K)]: DIPPR below 355 K, VDI above, VDI extrapolated past 478 K."""
    if T < 355.3:
        return dippr_eval("Cp_liquid", T) / MW / 1000
    Tv, cv = vdi_cp["Temperature_K"].to_numpy(), vdi_cp["Cp_l_kJkgK"].to_numpy()
    if T <= Tv[-1]:
        return np.interp(T, Tv, cv)
    return cv[-1] + (cv[-1] - cv[-2]) / (Tv[-1] - Tv[-2]) * (T - Tv[-1])


# ---------------------------------------------------------------- original file
orig = pd.read_csv(DATA_DIR / "IPA_properties.csv", skipinitialspace=True)
new = orig.copy()

# Phase of each row: Tsat is given on the first row of each pressure block
tsat = orig.groupby("Pressure_MPA")["TempVapor_Tsp_K"].transform("first")
T, P = orig["Temperature_K"], orig["Pressure_MPA"]
is_liquid = T < np.where(tsat < TC, tsat, TC)  # liquid below Tsat (subcritical) or below Tc (supercritical p)
is_vapour = (tsat < TC) & (T > tsat)           # vapour only exists at subcritical pressure

# 1. Liquid Cp
new.loc[is_liquid, "Cp_kJkgK"] = [liquid_cp(t) for t in T[is_liquid]]

# 2. Vapour viscosity and conductivity
new.loc[is_vapour, "Viscosity_mewPas"] = dippr_eval("mu_vapour", T[is_vapour]) * 1e6
new.loc[is_vapour, "ThermalConductivity_mWmK"] = dippr_eval("k_vapour", T[is_vapour]) * 1e3

# 3. Liquid density below the saturated-liquid density.
# Flag only where the value is below BOTH DIPPR and VDI saturated-liquid density. VDI starts at 355.65 K;
# below that, DIPPR alone sits ~1% above Celsius/Chemcast/the file, which is correlation scatter, not an error.
rho_sat = pd.Series(np.nan, index=orig.index)
rho_sat[is_liquid] = np.minimum(
    dippr_eval("rho_liquid", T[is_liquid]) * MW,
    np.interp(T[is_liquid], vdi["Temperature_K"], vdi["Density_l_kgm3"], left=np.nan, right=np.nan))
bad_rho = is_liquid & (orig["Density_kgm3"] < rho_sat)
for i in orig.index[bad_rho]:
    ref = cc_liq[P[i]]
    assert ref["temperature_K"].min() <= T[i] <= ref["temperature_K"].max()  # no extrapolation
    new.loc[i, "Density_kgm3"] = np.interp(T[i], ref["temperature_K"], ref["density_kgm3"])
ok = rho_sat.notna()
assert (new.loc[ok, "Density_kgm3"] >= rho_sat[ok]).all()


# ---------------------------------------------------------------- writers
# Same fixed-width formats as the original files (checked below by round-tripping the original data)
def write_csv(df):
    lines = [",".join(COLS)]
    for _, r in df.iterrows():
        f = [f"{r.iloc[0]:9.6f}", f"{r.iloc[1]:.2f}", f"{r.iloc[2]:.5f}", f"{r.iloc[3]:8.4f}",
             f"{r.iloc[4]:9.4f}", f"{r.iloc[5]:9.5f}"]
        if pd.notna(r.iloc[6]):
            f += [f"{r.iloc[6]:.2f}", f"{r.iloc[7]:.4f}"]
        else:
            f += [""]
        lines.append(",".join(f))
    return "\n".join(lines)


usr_text = (DATA_DIR / "usr_properties.txt").read_bytes().decode("utf-8")
usr_lines = usr_text.split("\r\n")
first_data = next(i for i, l in enumerate(usr_lines) if l.startswith("C3H8O")) + 1
n_rows = len(orig)


def write_usr(df):
    rows = []
    for _, r in df.iterrows():
        s = (f"{r.iloc[0]:10.6f}{r.iloc[1]:11.2f}{r.iloc[2]:17.5f}{r.iloc[3]:16.4f}"
             f"{r.iloc[4]:15.4f}{r.iloc[5]:15.5f}")
        if pd.notna(r.iloc[6]):
            s += f"{r.iloc[6]:12.2f}{r.iloc[7]:17.4f}"
        rows.append(s)
    out = usr_lines[:first_data] + rows + usr_lines[first_data + n_rows:]
    return "\r\n".join(out)


# The writers must reproduce the original files byte for byte from the original data
assert write_csv(orig) == (DATA_DIR / "IPA_properties.csv").read_text(), "CSV format mismatch"
assert write_usr(orig) == usr_text, "usr_properties format mismatch"

(OUT_DIR / "IPA_properties.csv").write_text(write_csv(new))
(OUT_DIR / "usr_properties.txt").write_bytes(write_usr(new).encode("utf-8"))


# ---------------------------------------------------------------- change log
reason = {"Cp_kJkgK": "liquid Cp -> DIPPR (<355 K) / VDI (>=355 K, extrapolated past 478 K)",
          "Viscosity_mewPas": "vapour viscosity -> DIPPR vapour correlation",
          "ThermalConductivity_mWmK": "vapour conductivity -> DIPPR vapour correlation",
          "Density_kgm3": "liquid density below sat. liquid -> Chemcast at same pressure"}
# Compare at the precision written to file
reread = pd.read_csv(OUT_DIR / "IPA_properties.csv", skipinitialspace=True)
log = []
for col in reason:
    changed = reread[col] != orig[col]
    for i in orig.index[changed]:
        log.append(dict(Pressure_MPA=P[i], Temperature_K=T[i], property=col, old=orig.loc[i, col],
                        new=reread.loc[i, col], change_pct=round(100 * (reread.loc[i, col] / orig.loc[i, col] - 1), 1),
                        reason=reason[col]))
log = pd.DataFrame(log)
log.to_csv(OUT_DIR / "changes.csv", index=False)

print(f"Rows: {n_rows}, values changed: {len(log)}")
print(log.groupby("property").size().to_string())
print(log.to_string(index=False))
