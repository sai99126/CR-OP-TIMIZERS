import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained pipeline and dataset
model_pipeline = joblib.load("xgb_model.pkl")
df = pd.read_excel("JHARKHAND_merged.xlsx")

# Features used during training
numeric_features = [
    'n_High','n_Medium','n_Low',
    'p_High','p_Medium','p_Low',
    'k_High','k_Medium','k_Low',
    'OC_High','OC_Medium','OC_Low',
    'pH_Alkaline','pH_Acidic','pH_Neutral',
    'EC_NonSaline','EC_Saline',
    'S_Sufficient','S_Deficient',
    'Fe_Sufficient','Fe_Deficient',
    'Zn_Sufficient','Zn_Deficient',
    'Cu_Sufficient','Cu_Deficient',
    'B_Sufficient','B_Deficient',
    'Mn_Sufficient','Mn_Deficient',
    'Temperature'
]
categorical_features = ['District', 'Soil Type', 'Crop']

st.title("🌱 Jharkhand Crop Recommendation System")

# --- User Inputs
district = st.selectbox("District", sorted(df["District"].unique()))
block_options = df[df["District"] == district]["Block"].dropna().unique()
block = st.selectbox("Block", sorted(block_options)) if len(block_options) > 0 else None

soil_options = df[(df["District"] == district) & (df["Block"] == block)]["Soil Texture"].dropna().unique()
soil_texture = st.selectbox("Soil Texture", sorted(soil_options)) if len(soil_options) > 0 else None

if st.button("🌱 Suitable Crops"):
    if soil_texture:
        # Filter a row for selected District/Block/Soil
        row = df[
            (df["District"] == district) &
            (df["Block"] == block) &
            (df["Soil Texture"] == soil_texture)
        ]

        if row.empty:
            st.warning("⚠ No matching data found for this selection.")
        else:
            # Use same preprocessing as training
            X_input = row[numeric_features + categorical_features].iloc[[0]]

            # Predict probabilities
            preds_proba = model_pipeline.predict_proba(X_input)[0]
            crop_classes = model_pipeline.classes_

            crop_probs = pd.DataFrame({
                "Crop": crop_classes,
                "Suitability (%)": preds_proba * 100
            }).sort_values(by="Suitability (%)", ascending=False).head(10)

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(crop_probs["Crop"], crop_probs["Suitability (%)"], color="green")
            ax.invert_yaxis()
            ax.set_xlabel("Suitability (%)")
            ax.set_title(f"Top 10 Predicted Suitable Crops\n{district} ({soil_texture})")

            for i, v in enumerate(crop_probs["Suitability (%)"]):
                ax.text(v + 0.5, i, f"{v:.2f}%", va="center")

            st.pyplot(fig)

            # Best crop
            best_crop = crop_probs.iloc[0]["Crop"]
            st.success(f"🌟 Best Recommended Crop: **{best_crop}**")

            # Show crop details
            details = df[df["Crop"] == best_crop][
                ["Season", "Category", "Crop", "Soil Type", "Soil Texture", "Sowing Time", "Spacing"]
            ].drop_duplicates()
            st.dataframe(details)
    else:
        st.warning("⚠ Please select a valid Soil Texture.")


