# app.py (robust version to avoid KeyError: '')
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium import CircleMarker, Popup, Tooltip

# ----------------- CONFIG -----------------
ARTIFACT_DIR = Path("artifacts")
CONFIG_PATH = ARTIFACT_DIR / "config.json"
DF_CLEAN_PATH = ARTIFACT_DIR / "df_clean_sample.csv"
RF_PATH = ARTIFACT_DIR / "model_rf.pkl"
KMEANS_PATH = ARTIFACT_DIR / "cluster_kmeans.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
CLUSTER_SCALER_PATH = ARTIFACT_DIR / "cluster_scaler.pkl"

st.set_page_config(layout="wide", page_title="Churn & Cluster Admin UI (Robust)")

# ----------------- HELPERS & LOADING -----------------
@st.cache_data(show_spinner=True)
def load_artifacts():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing {CONFIG_PATH}. Run training script first.")
    config = json.loads(CONFIG_PATH.read_text())

    if not DF_CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing {DF_CLEAN_PATH}. Run training script first.")
    df_snapshot = pd.read_csv(DF_CLEAN_PATH)

    if not (RF_PATH.exists() and KMEANS_PATH.exists() and SCALER_PATH.exists() and CLUSTER_SCALER_PATH.exists()):
        raise FileNotFoundError("One or more model/scaler artifacts missing in artifacts/. Run train_and_save.py first.")

    with open(RF_PATH, "rb") as f:
        rf_model = pickle.load(f)
    with open(KMEANS_PATH, "rb") as f:
        kmeans_model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler_model = pickle.load(f)
    with open(CLUSTER_SCALER_PATH, "rb") as f:
        cluster_scaler = pickle.load(f)

    return config, df_snapshot, rf_model, kmeans_model, scaler_model, cluster_scaler

try:
    config, df_snapshot, rf_model, kmeans_model, scaler_model, cluster_scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

ID_COL = config.get("id_col", "customer_id")
TARGET_COL = config.get("target_col", "churned")
FEATURE_COLS = config.get("feature_columns", [])
NUMERIC_FEATURES = config.get("numeric_features", [])

# Normalize index to string, strip whitespace, remove empty-string IDs
def prepare_df_for_lookup(df_raw, id_col=ID_COL):
    df = df_raw.copy()
    # If ID column present, move it to index after cleaning; else treat index as id
    if id_col in df.columns:
        # Convert to string and strip
        df[id_col] = df[id_col].astype(str).str.strip()
        # Identify empty IDs
        empty_id_mask = df[id_col].isna() | (df[id_col] == "") | (df[id_col].str.lower() == "nan")
        if empty_id_mask.any():
            # Keep snapshot for diagnostics but drop invalid rows from df used for lookup
            invalid_ids = df.loc[empty_id_mask, id_col].index.tolist()
            st.warning(f"Found {empty_id_mask.sum()} rows with empty or invalid '{id_col}' in the provided dataset. These rows will be ignored for lookups.")
            # show a tiny sample for debugging
            st.write(df.loc[empty_id_mask].head(5))
        df = df.loc[~empty_id_mask].copy()
        df = df.set_index(id_col)
        # ensure index is string and stripped
        df.index = df.index.astype(str).str.strip()
    else:
        # Data has no explicit id column; ensure index strings are clean
        df.index = df.index.astype(str).str.strip()
        # drop rows with empty-string index
        empty_index_mask = (df.index == "") | (df.index.str.lower() == "nan")
        if any(empty_index_mask):
            st.warning(f"Found {empty_index_mask.sum()} rows with empty index. These rows will be ignored for lookups.")
            df = df[~empty_index_mask].copy()
    return df

# ----------------- UI: File upload & user selection -----------------
st.title("Churn & Cluster Admin Dashboard — Robust")

col_upload, col_input = st.columns([2,1])
with col_upload:
    upload = st.file_uploader("Upload CSV (optional). If empty, snapshot will be used.", type=["csv"])
    if upload:
        try:
            df_input_raw = pd.read_csv(upload)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            st.stop()
    else:
        df_input_raw = df_snapshot.copy()

with col_input:
    user_id = st.text_input("Enter customer_id to inspect (string or numeric):")
    show_ids = st.checkbox("Show sample IDs (debug)", value=False)

# Prepare dataframe (clean empty IDs)
df_input = prepare_df_for_lookup(df_input_raw, ID_COL)

if show_ids:
    st.write("Sample IDs (first 30):")
    st.write(list(df_input.index[:30]))

# Validate user_id input
if user_id is None:
    user_id = ""
user_id_str = str(user_id).strip()

if user_id_str == "":
    st.error("You entered an empty customer_id (blank). Please type the customer_id and press Enter.")
    st.stop()

# Now check presence
if user_id_str not in df_input.index:
    st.error("User id not found in the current dataset. Possible causes:\n"
             "- The ID is not present in the uploaded CSV or snapshot.\n"
             "- Leading/trailing spaces or mismatched formatting. Use the debug checkbox to inspect sample IDs.\n"
             "Make sure the ID matches exactly after trimming spaces.")
    st.stop()

# Extract user row safely
try:
    user_row = df_input.loc[user_id_str]
except KeyError:
    st.error("Lookup failed: user id unexpectedly not found after normalization.")
    st.stop()

# ----------------- Ensure cluster assignment exists -----------------
cluster_cols = config.get("cluster_cols", [])
# if cluster cols missing, attempt to bring from snapshot for the selected user
missing_cluster_cols = [c for c in cluster_cols if c not in df_input.columns]
if missing_cluster_cols:
    st.info(f"Cluster columns missing in uploaded file: {missing_cluster_cols}. Attempting to retrieve values for the selected user from snapshot.")
    # attempt to find in snapshot
    for c in missing_cluster_cols:
        if c in df_snapshot.columns:
            try:
                # match by ID in snapshot; snapshot may not be indexed by id_col
                matching = df_snapshot[df_snapshot[ID_COL].astype(str).str.strip() == user_id_str]
                if not matching.empty:
                    user_row.at[c] = matching.iloc[0][c]
                    st.success(f"Retrieved {c} for user from snapshot.")
                else:
                    st.warning(f"No matching row for selected user in snapshot to fetch column {c}.")
            except Exception as e:
                st.warning(f"Could not retrieve {c} from snapshot: {e}")

# Final check for cluster cols presence for the user
missing_cluster_cols_final = [c for c in cluster_cols if c not in user_row.index and c not in df_input.columns]
if missing_cluster_cols_final:
    st.error("Insufficient cluster columns for computing user's cluster: " + ", ".join(missing_cluster_cols_final))
    st.stop()

# Compute or use existing cluster label
if "cluster" in df_input.columns:
    try:
        cluster_label = int(df_input.loc[user_id_str, "cluster"])
    except Exception:
        # fallback to compute from cluster cols
        cluster_vector = user_row[cluster_cols].values.reshape(1, -1).astype(float)
        cluster_vector_scaled = cluster_scaler.transform(cluster_vector)
        cluster_label = int(kmeans_model.predict(cluster_vector_scaled)[0])
else:
    cluster_vector = user_row[cluster_cols].values.reshape(1, -1).astype(float)
    try:
        cluster_vector_scaled = cluster_scaler.transform(cluster_vector)
        cluster_label = int(kmeans_model.predict(cluster_vector_scaled)[0])
    except Exception as e:
        st.error(f"Failed to compute cluster label for user: {e}")
        st.stop()

st.header(f"User `{user_id_str}` — Cluster {cluster_label}")

# ----------------- Build a profiling dataset (compute cluster for snapshot/upload) -----------------
df_profile = df_input.copy()
if "cluster" not in df_profile.columns:
    # compute clusters for rows that have all necessary cluster cols
    present_mask = df_profile[cluster_cols].notna().all(axis=1)
    if present_mask.sum() > 0:
        to_cluster = df_profile.loc[present_mask, cluster_cols].astype(float)
        to_cluster_scaled = cluster_scaler.transform(to_cluster.values)
        labels = kmeans_model.predict(to_cluster_scaled)
        df_profile.loc[present_mask, "cluster"] = labels
    else:
        st.warning("No rows in the dataset have all required cluster columns to compute cluster assignments for profiling.")

# Ensure target column numeric if present
if TARGET_COL in df_profile.columns:
    df_profile[TARGET_COL] = pd.to_numeric(df_profile[TARGET_COL], errors="coerce")

cluster_df = df_profile[df_profile["cluster"] == cluster_label].copy()
total_users = len(df_profile)
cluster_size = len(cluster_df)
cluster_churn_rate = cluster_df[TARGET_COL].mean() if TARGET_COL in cluster_df.columns else None
avg_watch = cluster_df["watch_hours"].mean() if "watch_hours" in cluster_df.columns else None
avg_login = cluster_df["last_login_days"].mean() if "last_login_days" in cluster_df.columns else None
avg_monthly = cluster_df["monthly_fee"].mean() if "monthly_fee" in cluster_df.columns else None

st.subheader("Cluster profile")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cluster size", f"{cluster_size}", delta=f"{(cluster_size/total_users*100 if total_users>0 else 0):.2f}% of dataset")
c2.metric("Cluster churn rate", f"{cluster_churn_rate:.2%}" if cluster_churn_rate is not None else "N/A")
c3.metric("Avg watch hours (cluster)", f"{avg_watch:.2f}" if avg_watch is not None else "N/A")
c4.metric("Avg last_login_days (cluster)", f"{avg_login:.2f}" if avg_login is not None else "N/A")
if avg_monthly is not None:
    st.write(f"Avg monthly fee (cluster): {avg_monthly:.2f}")

# ----------------- Churn prediction for the user -----------------
st.subheader("Churn prediction (RandomForest)")

# Build feature row aligned to FEATURE_COLS
X_user = pd.DataFrame(columns=FEATURE_COLS, index=[0])
for col in FEATURE_COLS:
    if col in user_row.index:
        X_user.at[0, col] = user_row[col]
    else:
        X_user.at[0, col] = 0

# Fill missing numeric features with snapshot medians
for num in NUMERIC_FEATURES:
    if num in X_user.columns:
        if pd.isna(X_user.at[0, num]) or X_user.at[0, num] == "":
            if num in df_snapshot.columns:
                X_user.at[0, num] = df_snapshot[num].median()
            else:
                X_user.at[0, num] = 0

# Scale numeric subset if possible
numeric_in_feature = [c for c in NUMERIC_FEATURES if c in X_user.columns]
if numeric_in_feature:
    try:
        X_user[numeric_in_feature] = scaler_model.transform(X_user[numeric_in_feature].astype(float))
    except Exception:
        st.warning("Feature scaling failed for numeric features; continuing without scaling for prediction.")
        X_user[numeric_in_feature] = X_user[numeric_in_feature].astype(float)

X_user = X_user.fillna(0)

# Predict
try:
    churn_proba = rf_model.predict_proba(X_user.values)[:, 1][0]
    st.write(f"Predicted probability of churn: **{churn_proba:.2%}**")
    st.progress(min(100, int(churn_proba * 100)))
except Exception as e:
    st.error(f"Failed to predict churn probability: {e}")

# ----------------- Map for cluster members -----------------
st.subheader("Geo map of users in the same cluster")

lat_candidates = [c for c in ["latitude", "lat"] if c in df_profile.columns]
lon_candidates = [c for c in ["longitude", "lon", "long"] if c in df_profile.columns]

if lat_candidates and lon_candidates:
    lat_col = lat_candidates[0]
    lon_col = lon_candidates[0]

    all_points = df_profile[[lat_col, lon_col]].dropna().copy()
    cluster_points = cluster_df[[lat_col, lon_col]].dropna().copy()

    if cluster_points.empty:
        st.info("No geo coordinates for users in this cluster.")
    else:
        mean_lat = float(cluster_points[lat_col].mean())
        mean_lon = float(cluster_points[lon_col].mean())
        fmap = folium.Map(location=[mean_lat, mean_lon], zoom_start=6, tiles="CartoDB positron")

        all_points["g_lat"] = all_points[lat_col].round(3)
        all_points["g_lon"] = all_points[lon_col].round(3)
        cluster_points["g_lat"] = cluster_points[lat_col].round(3)
        cluster_points["g_lon"] = cluster_points[lon_col].round(3)

        agg_cluster = cluster_points.groupby(["g_lat", "g_lon"]).size().reset_index(name="cluster_count")
        agg_all = all_points.groupby(["g_lat", "g_lon"]).size().reset_index(name="total_count")
        merged = agg_cluster.merge(agg_all, on=["g_lat", "g_lon"], how="left").fillna(0)

        total_points_all = len(all_points)
        for _, row in merged.iterrows():
            lat = float(row["g_lat"])
            lon = float(row["g_lon"])
            cluster_count = int(row["cluster_count"])
            total_count_loc = int(row["total_count"])
            pct_total = (total_count_loc / total_points_all * 100) if total_points_all > 0 else 0.0
            pct_cluster = (cluster_count / cluster_size * 100) if cluster_size > 0 else 0.0
            tooltip = f"{cluster_count} cluster users here — {total_count_loc} total users here"
            popup_html = f"""
            <b>Location:</b> {lat}, {lon}<br>
            <b>Cluster users here:</b> {cluster_count} ({pct_cluster:.2f}% of cluster)<br>
            <b>Total users here:</b> {total_count_loc} ({pct_total:.2f}% of dataset)
            """
            radius = 4 + (cluster_count ** 0.5) * 2
            CircleMarker(location=[lat, lon],
                         radius=radius,
                         fill=True,
                         fill_opacity=0.7,
                         popup=Popup(popup_html, max_width=300),
                         tooltip=Tooltip(tooltip)).add_to(fmap)

        if lat_col in user_row.index and lon_col in user_row.index and not pd.isna(user_row[lat_col]):
            try:
                CircleMarker(location=[float(user_row[lat_col]), float(user_row[lon_col])],
                             radius=8, color="red", fill=True, fill_opacity=1, tooltip="Selected user").add_to(fmap)
            except Exception:
                pass

        st_folium(fmap, width=900, height=500)
else:
    if "city" in df_profile.columns:
        st.info("No lat/lon columns. Showing city-level cluster distribution.")
        agg_city_cluster = cluster_df.groupby("city").size().reset_index(name="cluster_count")
        agg_city_all = df_profile.groupby("city").size().reset_index(name="total_count")
        merged_city = agg_city_cluster.merge(agg_city_all, on="city", how="left")
        merged_city["pct_cluster"] = merged_city["cluster_count"] / cluster_size * 100
        merged_city["pct_total"] = merged_city["total_count"] / total_users * 100
        st.dataframe(merged_city.sort_values("cluster_count", ascending=False).reset_index(drop=True))
    else:
        st.info("No geographic information (lat/lon or city) found in dataset. Add coordinates for geo visualizations.")

# ----------------- Other outputs -----------------
st.subheader("Cluster members sample & distribution")
col_a, col_b = st.columns([2,1])
with col_a:
    st.write("Sample rows from this cluster (top 20):")
    if not cluster_df.empty:
        st.dataframe(cluster_df.head(20))
    else:
        st.write("No members in this cluster in the provided dataset snapshot/upload.")
with col_b:
    st.write("Cluster distribution in dataset:")
    try:
        st.bar_chart(df_profile["cluster"].value_counts().sort_index())
    except Exception:
        st.info("Cluster column not present to build distribution chart.")

st.write("---")
st.caption("This app reads pretrained artifacts from the artifacts/ folder. If you update the training logic or add categorical features, re-run training to refresh artifacts.")
