# app_ledger_v1_5.py
import streamlit as st
import pandas as pd
import numpy as np
import hashlib, time, secrets, random, json
import matplotlib.pyplot as plt
import networkx as nx
from ecdsa import SigningKey, SECP256k1

st.set_page_config(page_title="CIT Ledger v1.5 – Annual Dynamics & Market Analytics", layout="wide")

st.title("🌍 CCS Carbon Integrity Token (CIT) Platform v1.5 – Dynamic Annual Simulation & Market Analytics")

# --------------------------- Sidebar Inputs ----------------------------------
st.sidebar.header("🔧 Base Project Parameters")
baseline_emissions = st.sidebar.number_input("Baseline CO₂ (tCO₂e/yr)", 10_000, 5_000_000, 1_000_000, 10_000)
capture_eff = st.sidebar.slider("Initial Capture Efficiency (%)", 50, 99, 90)
trans_leak = st.sidebar.slider("Initial Transport Leakage (%)", 0.0, 2.0, 1.5, 0.05)
stor_leak = st.sidebar.slider("Initial Storage Leakage (%)", 0.0, 2.0, 1.5, 0.05)
uncert = st.sidebar.slider("Initial Uncertainty (%)", 0.0, 10.0, 3.0, 0.5)
buffer_pct = st.sidebar.slider("Buffer Reserve (%)", 0.0, 10.0, 2.0, 0.5)
ins_cov = st.sidebar.slider("Insurance Coverage (%)", 0.0, 20.0, 10.0, 1.0)
ins_prem = st.sidebar.number_input("Insurance Premium ($/t)", 0.0, 5.0, 0.84, 0.1)
price = st.sidebar.number_input("Market Carbon Price ($/t)", 20.0, 300.0, 70.0, 1.0)
qual_prem = st.sidebar.slider("Quality Premium (%)", 0.0, 30.0, 10.0, 1.0)
risk_disc = st.sidebar.slider("Risk Discount (%)", 0.0, 20.0, 4.0, 0.5)
yrs = st.sidebar.slider("Project Duration (yrs)", 1, 10, 5)
disc_rate = st.sidebar.slider("Discount Rate (%)", 0.0, 20.0, 8.0, 0.5)
capex = st.sidebar.number_input("CAPEX ($M)", 1.0, 2000.0, 200.0, 10.0)*1_000_000
opex = st.sidebar.number_input("Annual OPEX ($M)", 0.1, 100.0, 10.0, 0.5)*1_000_000

# --------------------------- Initialize --------------------------------------
if "ledger" not in st.session_state:
    st.session_state.ledger = pd.DataFrame(columns=[
        "Year","Txn_Hash","Tonnes","Value_USD","Buyer_Segment","Price","BlockHash"])
if "registry" not in st.session_state:
    st.session_state.registry = []
if "priv_key" not in st.session_state:
    st.session_state.priv_key = SigningKey.generate(curve=SECP256k1)

# --------------------------- Annual Calculations -----------------------------
st.subheader("📅 Annual CO₂ Reductions and Financial Summary")

price_adj = price*(1+(qual_prem-risk_disc)/100)
annual_data=[]
curr_eff=capture_eff
curr_tleak=trans_leak
curr_sleak=stor_leak
curr_uncert=uncert

for yr in range(1,yrs+1):
    # Update parameters each year
    curr_eff=min(curr_eff+1,99)
    curr_tleak=max(curr_tleak-0.025,1)
    curr_sleak=max(curr_sleak-0.025,1)
    curr_uncert=max(curr_uncert-0.05,0.1)

    baseline = baseline_emissions
    project_em = baseline*(1-curr_eff/100) + baseline*((curr_tleak+curr_sleak)/100)
    reductions = baseline - project_em
    cons_red = reductions*(1-curr_uncert/100)
    buffer = cons_red*(buffer_pct/100)
    insured = cons_red - buffer
    insured_adj = insured*(1 - ins_cov/100)
    gross_rev = insured_adj*price_adj
    ins_cost = insured_adj*ins_prem
    net_rev = gross_rev - ins_cost

    annual_data.append([
        yr, baseline, project_em, reductions, insured_adj, gross_rev, ins_cost, net_rev,
        round(curr_eff,2), round(curr_tleak,3), round(curr_sleak,3), round(curr_uncert,3)
    ])

annual_df = pd.DataFrame(annual_data, columns=[
    "Year","Baseline_CO2","Project_CO2","Reductions","tCO2_Issued","Gross_$","Ins_Cost_$","Net_$",
    "Eff","Trans_Leak","Stor_Leak","Uncert"])

st.dataframe(annual_df.style.format({
    "Baseline_CO2": "{:,.0f}", "Project_CO2": "{:,.0f}",
    "Reductions": "{:,.0f}", "tCO2_Issued": "{:,.0f}",
    "Gross_$": "${:,.0f}", "Ins_Cost_$": "${:,.0f}", "Net_$": "${:,.0f}"
}))

# --------------------------- Annual Charts -----------------------------------
st.subheader("📊 Annual Performance Charts")

# (a) Baseline vs Project Emissions
fig1, ax1 = plt.subplots(figsize=(7,4))
ax1.bar(annual_df["Year"], annual_df["Baseline_CO2"], color="grey", alpha=0.6, label="Baseline")
ax1.bar(annual_df["Year"], annual_df["Project_CO2"], color="#27AE60", alpha=0.9, label="Project")
ax1.set_xlabel("Year"); ax1.set_ylabel("tCO₂e"); ax1.set_title("Baseline vs Project Emissions")
ax1.legend()
st.pyplot(fig1)

# (b) Revenue breakdown (Gross, Insurance, Net)
fig2, ax2 = plt.subplots(figsize=(7,4))
bar_width = 0.25
x = np.arange(len(annual_df))
ax2.bar(x - bar_width, annual_df["Gross_$"]/1e6, width=bar_width, color="#5DADE2", label="Gross")
ax2.bar(x, annual_df["Ins_Cost_$"]/1e6, width=bar_width, color="#E74C3C", label="Insurance")
ax2.bar(x + bar_width, annual_df["Net_$"]/1e6, width=bar_width, color="#27AE60", label="Net")
ax2.set_xticks(x)
ax2.set_xticklabels(annual_df["Year"])
ax2.set_xlabel("Year"); ax2.set_ylabel("Million USD")
ax2.set_title("Revenue Breakdown")
ax2.legend()
st.pyplot(fig2)

# --------------------------- Mint + Registry Mock ----------------------------
st.subheader("🪙 Mint CITs & Register with Buyer Attributes")

def fake_registry_post(entry):
    time.sleep(0.1)
    txid="VRR-"+hashlib.sha1(entry["Txn_Hash"].encode()).hexdigest()[:8]
    return {"registry_tx":txid,"status":"success"}

if st.button("Mint & Register Annual CITs"):
    parent="GENESIS"
    for _,row in annual_df.iterrows():
        ts=time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())
        payload=f"{ts}{row['Year']}{row['tCO2_Issued']}{price_adj}{parent}"
        txn_hash=hashlib.sha256(payload.encode()).hexdigest()
        sig=st.session_state.priv_key.sign(txn_hash.encode()).hex()[:64]
        block_hash=hashlib.sha256(f"{txn_hash}{sig}{secrets.token_hex(4)}".encode()).hexdigest()[:16]

        buyer=random.choice(["Corporate","Airline","Sovereign"])
        attrs={"Buyer_Segment":buyer,"VCMI_Tag":random.choice(["Gold","Silver"]),
               "CORSIA_Eligible":random.choice([True,False]),
               "Insurance_Tier":f"{ins_cov:.1f}%"}

        new_entry={
            "Year":int(row["Year"]),
            "Txn_Hash":txn_hash[:12],
            "Tonnes":round(row["tCO2_Issued"],0),
            "Value_USD":round(row["Net_$"],2),
            "Buyer_Segment":buyer,
            "Price":round(price_adj,2),
            "BlockHash":block_hash
        }
        st.session_state.ledger=pd.concat([st.session_state.ledger,pd.DataFrame([new_entry])],ignore_index=True)
        reg_resp=fake_registry_post(new_entry)
        st.session_state.registry.append({
            "Registry_TX":reg_resp["registry_tx"],"Txn_Hash":txn_hash[:12],
            "Buyer_Segment":buyer,"Attributes":attrs})
        parent=txn_hash
    st.success(f"✅ Minted and registered {yrs} annual CITs.")

# --------------------------- Ledger & Registry Display -----------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("📒 Ledger Summary")
    if len(st.session_state.ledger)>0:
        st.dataframe(st.session_state.ledger,use_container_width=True)
        st.metric("Total tCO₂ Issued",f"{st.session_state.ledger['Tonnes'].sum():,.0f}")
    else:
        st.info("No ledger entries yet – mint and register first.")
with col2:
    st.subheader("🌐 Registry Summary")
    if len(st.session_state.registry)>0:
        st.dataframe(pd.DataFrame(st.session_state.registry),use_container_width=True)
    else:
        st.info("Registry empty – mint first.")

# --------------------------- Interactive Market Analytics --------------------
st.subheader("💹 Interactive Market Analytics Dashboard")

if len(st.session_state.ledger)>0:
    ledger = st.session_state.ledger.copy()
    buyer_filter = st.multiselect(
        "Select Buyer Segments", options=["Corporate","Airline","Sovereign"],
        default=["Corporate","Airline","Sovereign"]
    )
    price_adj_factor = st.slider("Adjust Market Price Multiplier (x)", 0.5, 2.0, 1.0, 0.1)
    disc = st.slider("Discount Rate for NPV (%)", 0.0, 15.0, 8.0, 0.5)

    filtered = ledger[ledger["Buyer_Segment"].isin(buyer_filter)]
    filtered["Adj_Value"] = filtered["Value_USD"] * price_adj_factor
    filtered["PV"] = filtered["Adj_Value"] / ((1 + disc/100) ** filtered["Year"])
    npv = filtered["PV"].sum() - capex
    st.metric("Portfolio NPV ($M)", f"{npv/1e6:,.2f}")

    # Buyer Segmentation Plot
    seg = filtered.groupby("Buyer_Segment")["Tonnes"].sum().reset_index()
    fig3, ax3 = plt.subplots(figsize=(6,4))
    ax3.bar(seg["Buyer_Segment"], seg["Tonnes"], color="#9b59b6")
    ax3.set_ylabel("tCO₂e")
    ax3.set_title("CIT Distribution by Buyer Segment")
    st.pyplot(fig3)
else:
    st.info("No data yet – mint and register first.")

# --------------------------- Traceability Graph ------------------------------
st.subheader("🔗 Transaction Trace Graph")
if len(st.session_state.ledger)>0:
    G=nx.DiGraph()
    for _,r in st.session_state.ledger.iterrows():
        G.add_edge(f"Y{r['Year']-1}", f"Y{r['Year']}", weight=r["Tonnes"])
    pos=nx.spring_layout(G,seed=42)
    fig,ax=plt.subplots(figsize=(7,4))
    nx.draw(G,pos,with_labels=True,node_color="#AED6F1",node_size=2200,
            font_size=9,arrowsize=15,ax=ax)
    st.pyplot(fig)
else:
    st.info("No graph yet.")

# --------------------------- Footer ------------------------------------------
st.markdown("---")
st.caption("""
**v1.5 Enhancements**
- 📈 Annual variability: capture +1%/yr (≤99%), leakage −0.025%/yr (≥1%), uncertainty −0.05%/yr (≥0.1%).  
- 📊 Added charts: (a) Baseline vs Project Emissions (b) Revenue Breakdown.  
- 💹 Added Interactive Market Analytics with buyer filters, price slider, dynamic NPV.  
- 🧾 Portfolio summary + segmentation visualizations.  
""")
