"""
frontend/app.py
-----------------
Streamlit UI for PropLens rent prediction + yield/investment scoring.

Run with:
    streamlit run frontend/app.py

Expects the FastAPI backend running at http://localhost:8000
(start it separately with: uvicorn api.main:app --reload --port 8000)
"""

import requests
import streamlit as st

API_BASE = "http://localhost:8000"
AED_PER_USD = 3.6725  # fixed peg since 1997, safe to hardcode

st.set_page_config(page_title="PropLens - Rent & Yield", layout="centered")
st.title("PropLens: Rent Prediction & Investment Scoring")

currency = st.radio("Currency", ["AED", "USD"], horizontal=True)


def to_display(usd_amount: float) -> float:
    return usd_amount * AED_PER_USD if currency == "AED" else usd_amount


def to_usd(display_amount: float) -> float:
    return display_amount / AED_PER_USD if currency == "AED" else display_amount


def fmt(usd_amount: float) -> str:
    symbol = "AED" if currency == "AED" else "$"
    return f"{symbol} {to_display(usd_amount):,.0f}"


tab1, tab2, tab3 = st.tabs(["Rent Prediction", "Yield & Investment Score", "Property Price Prediction"])


@st.cache_data(ttl=300)
def get_communities():
    try:
        resp = requests.get(f"{API_BASE}/communities", timeout=5)
        resp.raise_for_status()
        return resp.json()["communities"]
    except Exception as e:
        st.error(f"Could not load communities from API: {e}")
        return []


communities = get_communities()
community_names = sorted({c["community"] for c in communities}) if communities else []


@st.cache_data(ttl=300)
def get_price_areas():
    try:
        resp = requests.get(f"{API_BASE}/price-areas", timeout=5)
        resp.raise_for_status()
        return resp.json()["areas"]
    except Exception as e:
        st.error(f"Could not load price areas from API: {e}")
        return []


price_areas = get_price_areas()

with tab1:
    st.subheader("Predict rent per sqft for a community")
    if community_names:
        community = st.selectbox("Community", community_names, key="rent_community")
    else:
        community = st.text_input("Community", key="rent_community_txt")
    size_sqft = st.number_input("Size (sqft)", min_value=100, value=1000, step=50, key="rent_size")

    if st.button("Predict Rent"):
        payload = {"community": community, "size_sqft": size_sqft}
        try:
            resp = requests.post(f"{API_BASE}/predict-rent", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            st.success(f"Estimated annual rent: {fmt(data['estimated_annual_rent_usd'])}")
            st.write(f"Rent per sqft: {fmt(data['rent_per_sqft_annual_usd'])}/sqft/year")
            st.caption(f"Based on {data['n_comparables']} comparable(s) -- source: {data['source']}")
        except requests.exceptions.RequestException as e:
            st.error(f"API error: {e}")

with tab2:
    st.subheader("Compute yield, ROI, and investment score")
    if community_names:
        community2 = st.selectbox("Community", community_names, key="yield_community")
    else:
        community2 = st.text_input("Community", key="yield_community_txt")

    col1, col2 = st.columns(2)
    with col1:
        size_sqft2 = st.number_input("Size (sqft)", min_value=100, value=1000, step=50, key="yield_size")
        price_label = f"Purchase price ({currency})"
        default_price = 500000 * AED_PER_USD if currency == "AED" else 500000
        purchase_price_input = st.number_input(price_label, min_value=1000.0, value=float(default_price), step=5000.0)
        purchase_price = to_usd(purchase_price_input)  # API always expects USD
        holding_years = st.slider("Holding period (years)", 1, 10, 5)
    with col2:
        service_charge = st.slider("Service charge (% of rent)", 0.0, 0.3, 0.12)
        vacancy = st.slider("Expected vacancy (%)", 0.0, 0.2, 0.05)
        mgmt_fee = st.slider("Management fee (% of rent)", 0.0, 0.15, 0.05)

    if st.button("Compute Yield & Score"):
        payload = {
            "community": community2,
            "size_sqft": size_sqft2,
            "purchase_price_usd": purchase_price,
            "holding_period_years": holding_years,
            "service_charge_pct": service_charge,
            "vacancy_pct": vacancy,
            "management_fee_pct": mgmt_fee,
        }
        try:
            resp = requests.post(f"{API_BASE}/yield-score", json=payload, timeout=10)
            resp.raise_for_status()
            r = resp.json()

            c1, c2, c3 = st.columns(3)
            c1.metric("Gross Yield", f"{r['gross_yield_pct']}%")
            c2.metric("Net Yield", f"{r['net_yield_pct']}%")
            c3.metric("Investment Score", f"{r['investment_score']}/100")

            st.write(f"**Estimated annual rent:** {fmt(r['predicted_annual_rent_usd'])}")
            st.write(f"**{holding_years}-year ROI (rental income only):** {r['roi_5yr_pct']}%")
            st.write(f"**Confidence:** {r['confidence']} ({r['n_comparables']} comparables)")

            with st.expander("Score breakdown"):
                st.json(r["score_breakdown"])
            if r["notes"]:
                for n in r["notes"]:
                    st.caption(f"Note: {n}")
        except requests.exceptions.RequestException as e:
            st.error(f"API error: {e}")

with tab3:
    st.subheader("Predict property sale price (DLD-based)")
    st.caption("Note: this AED figure is not USD-converted -- Dubai property prices are "
               "modeled and quoted directly in AED, since that's the DLD transaction currency.")

    col1, col2 = st.columns(2)
    with col1:
        if price_areas:
            price_area = st.selectbox("Area name", price_areas, key="price_area")
        else:
            price_area = st.text_input("Area name", value="Al Wasl", key="price_area_txt")
        sub_type = st.selectbox("Property sub-type", ["Flat", "Villa", "Townhouse"], key="price_subtype")
        area_sqm = st.number_input("Size (sqm)", min_value=15.0, max_value=1500.0, value=90.0, step=5.0)
        bedrooms = st.number_input("Bedrooms (0 = studio)", min_value=0, max_value=10, value=2)
    with col2:
        p_year = st.number_input("Transaction year", min_value=2007, max_value=2027, value=2026)
        p_month = st.slider("Month", 1, 12, 6)
        has_parking2 = st.checkbox("Has parking", value=True)
        is_offplan2 = st.checkbox("Off-plan", value=False)
        near_landmark2 = st.checkbox("Near a notable landmark", value=False)
        near_mall2 = st.checkbox("Near a mall", value=False)
        near_metro2 = st.checkbox("Near a metro station", value=False)

    if st.button("Predict Price"):
        payload = {
            "area_name": price_area, "property_sub_type": sub_type,
            "area_sqm": area_sqm, "bedrooms": bedrooms,
            "has_parking": has_parking2, "is_offplan": is_offplan2,
            "near_landmark": near_landmark2, "near_mall": near_mall2, "near_metro": near_metro2,
            "year": p_year, "month": p_month,
        }
        try:
            resp = requests.post(f"{API_BASE}/predict-price", json=payload, timeout=15)
            resp.raise_for_status()
            r = resp.json()

            st.success(f"Predicted price: AED {r['calibrated_predicted_price']:,.0f}")
            st.caption(f"Raw model output: AED {r['raw_predicted_price']:,.0f} "
                       f"(calibration factor: {r['calibration_factor']}x, adjusts for market "
                       f"trend beyond the training data's date range)")

            conf_color = {"high": "green", "medium": "orange", "low": "red"}.get(r["confidence"], "gray")
            st.markdown(f"**Confidence:** :{conf_color}[{r['confidence']}]")
            for n in r.get("notes", []):
                st.warning(n)

            st.write("**Top price drivers (SHAP):**")
            for d in r["top_price_drivers"]:
                sign = "+" if d["impact_aed"] > 0 else ""
                st.write(f"- `{d['feature']}`: {sign}AED {d['impact_aed']:,.0f} ({d['direction']})")

            with st.expander("Full response"):
                st.json(r)
        except requests.exceptions.RequestException as e:
            st.error(f"API error: {e}")

st.divider()
st.caption("Yield is computed primarily from real market comparables for the matched "
           "community, with the ML model used as a fallback when comparables are thin. "
           "See src/yield_engine.py for the exact blending logic. Price prediction is "
           "trained on real DLD sales transactions (Sales only, Mortgages/Gifts excluded); "
           "see src/price/ for the full pipeline and known limitations.")