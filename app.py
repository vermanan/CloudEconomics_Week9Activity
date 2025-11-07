import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------
st.set_page_config(page_title="AWS EC2 & S3 EDA Dashboard", layout="wide")
st.title("☁️ AWS EC2 & S3 Exploratory Data Analysis (EDA)")
st.markdown("""
### INFO49971 – Cloud Economics (Week 9 Activity)
This dashboard performs **comparative exploratory data analysis (EDA)** on AWS **EC2** and **S3** resources.
You can explore usage patterns, cost distributions, and identify optimization opportunities.
""")

# -----------------------------------------------------
# File Upload
# -----------------------------------------------------
ec2_df = pd.read_csv("aws_resources_compute.csv")
s3_df = pd.read_csv("aws_resources_S3.csv")

# -----------------------------------------------------
# Sidebar Filters
# -----------------------------------------------------
st.sidebar.header("🔍 Filters")

# --- EC2 Filters ---
st.sidebar.subheader("🖥️ EC2 Filters")
regions_ec2 = ec2_df["Region"].unique() if "Region" in ec2_df.columns else []
types_ec2 = ec2_df["InstanceType"].unique() if "InstanceType" in ec2_df.columns else []
states_ec2 = ec2_df["State"].unique() if "State" in ec2_df.columns else []

region_filter_ec2 = st.sidebar.multiselect("Select EC2 Regions", regions_ec2, default=list(regions_ec2))
type_filter_ec2 = st.sidebar.multiselect("Select Instance Types", types_ec2, default=list(types_ec2))
state_filter_ec2 = st.sidebar.multiselect("Select States", states_ec2, default=list(states_ec2))

filtered_ec2 = ec2_df.copy()
if "Region" in ec2_df.columns:
    filtered_ec2 = filtered_ec2[filtered_ec2["Region"].isin(region_filter_ec2)]
if "InstanceType" in ec2_df.columns:
    filtered_ec2 = filtered_ec2[filtered_ec2["InstanceType"].isin(type_filter_ec2)]
if "State" in ec2_df.columns:
    filtered_ec2 = filtered_ec2[filtered_ec2["State"].isin(state_filter_ec2)]

# --- S3 Filters ---
st.sidebar.subheader("🪣 S3 Filters")
regions_s3 = s3_df["Region"].unique() if "Region" in s3_df.columns else []
storage_classes = s3_df["StorageClass"].unique() if "StorageClass" in s3_df.columns else []
encryptions = s3_df["Encryption"].unique() if "Encryption" in s3_df.columns else []

region_filter_s3 = st.sidebar.multiselect("Select S3 Regions", regions_s3, default=list(regions_s3))
class_filter_s3 = st.sidebar.multiselect("Select Storage Classes", storage_classes, default=list(storage_classes))
encrypt_filter_s3 = st.sidebar.multiselect("Select Encryption Type", encryptions, default=list(encryptions))

filtered_s3 = s3_df.copy()
if "Region" in s3_df.columns:
    filtered_s3 = filtered_s3[filtered_s3["Region"].isin(region_filter_s3)]
if "StorageClass" in s3_df.columns:
    filtered_s3 = filtered_s3[filtered_s3["StorageClass"].isin(class_filter_s3)]
if "Encryption" in s3_df.columns:
    filtered_s3 = filtered_s3[filtered_s3["Encryption"].isin(encrypt_filter_s3)]

# --- File Uploads ---
ec2_file = st.sidebar.file_uploader("Upload EC2 Dataset (CSV)", type=["csv"])
s3_file = st.sidebar.file_uploader("Upload S3 Dataset (CSV)", type=["csv"])

if ec2_file is not None:
    ec2_df = pd.read_csv(ec2_file)
    st.sidebar.success("✅ EC2 dataset loaded.")
else:
    st.sidebar.info("Using default file: aws_resources_compute.csv")

if s3_file is not None:
    s3_df = pd.read_csv(s3_file)
    st.sidebar.success("✅ S3 dataset loaded.")
else:
    st.sidebar.info("Using default file: aws_resources_S3.csv")

# -----------------------------------------------------
# Tabs for EC2, S3, and Optimization
# -----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🖥️ EC2 Analysis",
    "🪣 S3 Analysis",
    "🚀 Optimization Suggestions",
    "🤖 ML Cost Optimization"
])

# =====================================================
# 🖥️ TAB 1: EC2 Analysis
# =====================================================
with tab1:
    # -----------------------------------------------------
    # EC2 Dataset Overview
    # -----------------------------------------------------
    st.header("🖥️ EC2 Instance Analysis")

    st.subheader("📊 Dataset Overview (EC2)")
    st.write(f"Filtered EC2 Instances: **{len(filtered_ec2)}**")
    st.dataframe(filtered_ec2.head())

    with st.expander("View EC2 Summary Info"):
        st.write(filtered_ec2.describe(include="all"))
        st.write("Missing values per column:")
        st.write(filtered_ec2.isna().sum())

    # -----------------------------------------------------
    # EC2 Visualizations
    # -----------------------------------------------------
    st.subheader("📈 EC2 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        if "CPUUtilization" in filtered_ec2.columns:
            st.write("**CPU Utilization Histogram**")
            fig_ec2_hist = px.histogram(filtered_ec2, x="CPUUtilization", nbins=20, title="EC2 CPU Utilization Distribution")
            st.plotly_chart(fig_ec2_hist, use_container_width=True)

    # -----------------------------------------------------
    # 💹 EC2: CPU Utilization vs Hourly Cost (Aligned Layout)
    # -----------------------------------------------------

    st.subheader("💹 EC2: CPU Utilization vs Hourly Cost")

    # Auto-detect possible column names
    def find_column(df, keywords):
        """Find the first column in df whose name matches any keyword."""
        for col in df.columns:
            lower = col.lower().replace(" ", "")
            if any(k in lower for k in keywords):
                return col
        return None

    cpu_col = find_column(filtered_ec2, ["cpuutil", "cpuusage", "avgcpu"])
    cost_col = find_column(filtered_ec2, ["costperhour", "hourlycost", "costusd", "priceusd", "billing"])
    region_col = find_column(filtered_ec2, ["region"])
    instance_col = find_column(filtered_ec2, ["instancetype"])
    id_col = find_column(filtered_ec2, ["instanceid"])

    if cpu_col and cost_col:
        # Use equal-width columns for clean side-by-side alignment
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### 📋 Preview of EC2 Instances (Top 10)")
            preview_cols = [c for c in [id_col, instance_col, region_col, cpu_col, cost_col] if c]
            st.dataframe(
                filtered_ec2[preview_cols].head(10),
                use_container_width=True,
                height=360
            )

        with col2:
            st.markdown("#### 📈 EC2 CPU Utilization vs Hourly Cost (USD)")
            fig_ec2_scatter = px.scatter(
                filtered_ec2,
                x=cpu_col,
                y=cost_col,
                color=region_col if region_col else None,
                hover_data=[id_col, instance_col, region_col] if id_col and instance_col and region_col else None,
                title="",
                size=cost_col,
                size_max=10,
                template="plotly_white"
            )
            fig_ec2_scatter.update_layout(
                height=400,
                margin=dict(l=30, r=30, t=30, b=30),
                legend_title_text="Region"
            )
            st.plotly_chart(fig_ec2_scatter, use_container_width=True)
    else:
        st.warning("⚠️ Could not find both CPU Utilization and Cost columns in EC2 dataset.")



    # -----------------------------------------------------
    # 💰 Top 5 Most Expensive EC2 Instances (Side-by-Side)
    # -----------------------------------------------------
    def find_column(df, keywords):
        for col in df.columns:
            lower = col.lower().replace(" ", "")
            if any(k in lower for k in keywords):
                return col
        return None

    cost_col = find_column(filtered_ec2, ["costperhour", "hourlycost", "costusd", "priceusd", "billing"])
    instance_col = find_column(filtered_ec2, ["instancetype", "instancetypeid", "type"])
    region_col = find_column(filtered_ec2, ["region"])
    id_col = find_column(filtered_ec2, ["instanceid"])

    if cost_col:
        st.subheader("💰 Top 5 Most Expensive EC2 Instances")

        # Select top 5 instances
        top5_ec2 = filtered_ec2.nlargest(5, cost_col)
        display_cols = [c for c in [id_col, instance_col, region_col, cost_col] if c]

        # Create two equal columns for layout
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### 📋 Instance Cost Overview (Top 5)")
            st.dataframe(
                top5_ec2[display_cols],
                use_container_width=True,
                height=320
            )

        with col2:
            st.markdown("#### 📊 Cost per Instance (USD/hour)")
            fig_top5_ec2 = px.bar(
                top5_ec2,
                x=id_col if id_col else instance_col,
                y=cost_col,
                color=region_col if region_col else None,
                hover_data=display_cols,
                text_auto=True,
                template="plotly_white"
            )
            fig_top5_ec2.update_layout(
                height=360,
                margin=dict(l=30, r=30, t=30, b=30),
                legend_title_text="Region"
            )
            st.plotly_chart(fig_top5_ec2, use_container_width=True)

    else:
        st.warning("⚠️ No cost column found in EC2 dataset — cannot compute top 5 expensive instances.")


    # -----------------------------------------------------
    # 🌍 Average EC2 Cost per Region (Side-by-Side Layout)
    # -----------------------------------------------------
    def find_column(df, keywords):
        for col in df.columns:
            lower = col.lower().replace(" ", "")
            if any(k in lower for k in keywords):
                return col
        return None

    cost_col = find_column(filtered_ec2, ["costperhour", "hourlycost", "costusd", "priceusd", "billing"])
    region_col = find_column(filtered_ec2, ["region"])

    if cost_col and region_col:
        st.subheader("🌍 Average EC2 Cost per Region")

        # Compute averages
        avg_ec2_cost = (
            filtered_ec2.groupby(region_col)[cost_col]
            .mean()
            .reset_index()
            .sort_values(cost_col, ascending=False)
        )

        # Create two side-by-side columns
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### 📋 Average Hourly Cost by Region")
            st.dataframe(
                avg_ec2_cost.style.format({cost_col: "{:.4f}"}),
                use_container_width=True,
                height=320
            )

        with col2:
            st.markdown("#### 📊 Regional Cost Comparison (USD/hour)")
            fig_ec2_cost = px.bar(
                avg_ec2_cost,
                x=region_col,
                y=cost_col,
                color=region_col,
                text_auto=".2f",
                template="plotly_white"
            )
            fig_ec2_cost.update_layout(
                height=360,
                margin=dict(l=30, r=30, t=30, b=30),
                showlegend=False
            )
            st.plotly_chart(fig_ec2_cost, use_container_width=True)

    else:
        st.warning("⚠️ Could not find Region or Cost column in EC2 dataset.")


# =====================================================
# 🪣 TAB 2: S3 Analysis
# =====================================================
with tab2:
    st.header("🪣 S3 Bucket Analysis")
    st.subheader("📊 Dataset Overview (S3)")
    st.write(f"Filtered S3 Buckets: **{len(filtered_s3)}**")
    st.dataframe(filtered_s3.head())

    with st.expander("View S3 Summary Info"):
        st.write(filtered_s3.describe(include="all"))
        st.write("Missing values per column:")
        st.write(filtered_s3.isna().sum())

    st.subheader("📈 S3 Visualizations")
    possible_cost_cols = [c for c in filtered_s3.columns if any(x in c.lower() for x in ["cost", "price", "usd", "billing"])]
    cost_col = possible_cost_cols[0] if possible_cost_cols else None

    col3, col4 = st.columns(2)
    with col3:
        if all(col in filtered_s3.columns for col in ["Region", "TotalSizeGB"]):
            st.write("**Total Storage by Region**")
            storage_region = filtered_s3.groupby("Region")["TotalSizeGB"].sum().reset_index()
            fig_s3_storage = px.bar(storage_region, x="Region", y="TotalSizeGB", title="Total S3 Storage by Region (GB)")
            st.plotly_chart(fig_s3_storage, use_container_width=True)

    with col4:
        if "TotalSizeGB" in filtered_s3.columns and cost_col:
            st.write("**Cost vs Storage Scatter Plot**")
            fig_s3_scatter = px.scatter(
                filtered_s3,
                x="TotalSizeGB",
                y=cost_col,
                color="StorageClass" if "StorageClass" in filtered_s3.columns else None,
                hover_data=["BucketName", "Region"],
                title=f"S3 {cost_col} vs Storage Size"
            )
            st.plotly_chart(fig_s3_scatter, use_container_width=True)

    # -----------------------------------------------------
    # 🪣 Top 5 Largest S3 Buckets (Side-by-Side Layout)
    # -----------------------------------------------------
    if "TotalSizeGB" in filtered_s3.columns:
        st.subheader("🪣 Top 5 Largest S3 Buckets")

        # Get top 5 buckets by size
        top5_s3 = filtered_s3.nlargest(5, "TotalSizeGB")

        display_cols = ["BucketName", "Region", "TotalSizeGB"]
        if cost_col:
            display_cols.append(cost_col)

        # Create two columns side by side
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### 📋 Largest Buckets Overview (Top 5)")
            st.dataframe(
                top5_s3[display_cols],
                use_container_width=True,
                height=320
            )

        with col2:
            st.markdown("#### 📊 Storage Size by Bucket (GB)")
            fig_top5_s3 = px.bar(
                top5_s3,
                x="BucketName",
                y="TotalSizeGB",
                color="Region",
                text_auto=True,
                template="plotly_white",
                title=""
            )
            fig_top5_s3.update_layout(
                height=360,
                margin=dict(l=30, r=30, t=30, b=30),
                legend_title_text="Region"
            )
            st.plotly_chart(fig_top5_s3, use_container_width=True)

    else:
        st.warning("⚠️ Column 'TotalSizeGB' not found in dataset.")


# =====================================================
# 🚀 TAB 3: Optimization Suggestions
# =====================================================
with tab3:
    st.header("🚀 Optimization Suggestions")

    # -----------------------------------------------------
    # 🖥️ EC2 Optimization
    # -----------------------------------------------------
    st.markdown("""
    ### 🖥️ EC2 Optimization
    - Identify underutilized instances (CPU < 20%) and consider stopping or resizing them.
    - Purchase **Savings Plans** or **Reserved Instances** for workloads that run continuously.
    - Consider using **Spot Instances** for short-term workloads.
    """)

    st.write("### 📊 EC2 Optimization Insights")

    if "CPUUtilization" in filtered_ec2.columns:
        # Add dummy cost if missing
        if "CostPerHourUSD" not in filtered_ec2.columns:
            filtered_ec2["CostPerHourUSD"] = 0.05

        # Categorize utilization
        ec2_opt = filtered_ec2.copy()
        ec2_opt["UtilizationCategory"] = ec2_opt["CPUUtilization"].apply(
            lambda x: "Underutilized (<20%)" if x < 20 
            else ("Optimal (20–70%)" if x < 70 else "High Utilization (≥70%)")
        )
        ec2_opt["SavingsPlanCandidate"] = ec2_opt["CPUUtilization"].apply(
            lambda x: "Always-On (Candidate)" if x >= 70 else "Variable"
        )
        ec2_opt["SpotCandidate"] = ec2_opt["CPUUtilization"].apply(
            lambda x: "Spot Candidate (<20%)" if x < 20 else "Regular"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            underutil_summary = ec2_opt["UtilizationCategory"].value_counts().reset_index()
            underutil_summary.columns = ["Category", "Count"]
            fig1 = px.pie(
                underutil_summary,
                names="Category",
                values="Count",
                hole=0.45,
                title="🧠 CPU Utilization Levels",
                template="plotly_white",
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            savings_summary = ec2_opt["SavingsPlanCandidate"].value_counts().reset_index()
            savings_summary.columns = ["Category", "Count"]
            fig2 = px.bar(
                savings_summary,
                x="Category",
                y="Count",
                color="Category",
                text_auto=True,
                title="💰 Savings Plan Candidates",
                template="plotly_white",
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            spot_summary = ec2_opt["SpotCandidate"].value_counts().reset_index()
            spot_summary.columns = ["Category", "Count"]
            fig3 = px.bar(
                spot_summary,
                x="Category",
                y="Count",
                color="Category",
                text_auto=True,
                title="⚡ Spot Instance Opportunities",
                template="plotly_white",
            )
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("⚠️ Could not find 'CPUUtilization' column in EC2 dataset.")

    # -----------------------------------------------------
    # 🪣 S3 Optimization
    # -----------------------------------------------------
    st.markdown("""
    ### 🪣 S3 Optimization
    - Move infrequently accessed data to cheaper storage classes (e.g., **S3 Standard-IA**, **S3 Glacier**).
    - Enable **lifecycle policies** to automatically delete old or unused objects.
    - Turn on **S3 Intelligent-Tiering** for automatic cost optimization.
    """)

    st.write("### 📊 S3 Optimization Insights")

    if "StorageClass" in filtered_s3.columns and "TotalSizeGB" in filtered_s3.columns:
        s3_opt = filtered_s3.copy()

        # Assign storage class category summaries
        storage_summary = s3_opt["StorageClass"].value_counts().reset_index()
        storage_summary.columns = ["StorageClass", "Count"]

        # Example of lifecycle age column (simulate)
        if "ObjectAgeDays" not in s3_opt.columns:
            s3_opt["ObjectAgeDays"] = (s3_opt.index % 300) + 1
        s3_opt["LifecyclePolicyEligible"] = s3_opt["ObjectAgeDays"].apply(
            lambda x: "Eligible (>180 days)" if x > 180 else "Recent (<180 days)"
        )
        lifecycle_summary = s3_opt["LifecyclePolicyEligible"].value_counts().reset_index()
        lifecycle_summary.columns = ["Category", "Count"]

        # Intelligent Tiering proportion (simulate if not available)
        s3_opt["IntelligentTiering"] = s3_opt["StorageClass"].apply(
            lambda x: "Intelligent-Tiering" if "Intelligent" in x else "Other"
        )
        tiering_summary = s3_opt["IntelligentTiering"].value_counts().reset_index()
        tiering_summary.columns = ["Category", "Count"]

        col4, col5, col6 = st.columns(3)

        with col4:
            fig4 = px.pie(
                storage_summary,
                names="StorageClass",
                values="Count",
                hole=0.45,
                title="🪣 Storage Class Distribution",
                template="plotly_white",
            )
            st.plotly_chart(fig4, use_container_width=True)

        with col5:
            fig5 = px.bar(
                lifecycle_summary,
                x="Category",
                y="Count",
                color="Category",
                text_auto=True,
                title="⏳ Lifecycle Policy Eligible Objects",
                template="plotly_white",
            )
            st.plotly_chart(fig5, use_container_width=True)

        with col6:
            fig6 = px.bar(
                tiering_summary,
                x="Category",
                y="Count",
                color="Category",
                text_auto=True,
                title="🤖 Intelligent-Tiering Usage",
                template="plotly_white",
            )
            st.plotly_chart(fig6, use_container_width=True)
    else:
        st.warning("⚠️ S3 dataset missing required columns ('StorageClass' or 'TotalSizeGB').")



# # =====================================================
# # 🤖 TAB 4: Machine Learning Cost Optimization Model
# # =====================================================
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, r2_score

# with tab4:
#     st.header("🤖 Machine Learning Model – EC2 Cost Prediction")
#     st.markdown("""
#     This section uses a simple **Linear Regression** model to predict the **hourly cost (USD)** 
#     of EC2 instances based on performance metrics such as CPU utilization, memory, or storage.
#     It helps identify which metrics most influence cost, and estimate cost for new configurations.
#     """)

#     # --- Helper function to detect relevant columns ---
#     def find_column(df, keywords):
#         for col in df.columns:
#             if any(k.lower() in col.lower().replace(" ", "") for k in keywords):
#                 return col
#         return None

#     # --- Automatically detect the key columns ---
#     cpu_col = find_column(filtered_ec2, ["cpuutil", "cpuusage", "cpuutilization"])
#     cost_col = find_column(filtered_ec2, ["costperhour", "hourlycost", "costusd", "priceusd"])

#     # --- Display detected columns ---
#     st.write("📋 Columns detected in EC2 dataset:", filtered_ec2.columns.tolist())

#     if cpu_col and cost_col:
#         st.success(f"✅ Detected columns: CPU → **{cpu_col}**, Cost → **{cost_col}**")

#         # --- Select numeric columns for model training ---
#         numeric_cols = filtered_ec2.select_dtypes(include=["float64", "int64"]).columns.tolist()
#         feature_cols = [col for col in numeric_cols if col != cost_col]

#         if len(feature_cols) == 0:
#             st.warning("⚠️ No numeric columns found to train the model.")
#         else:
#             # Prepare training data
#             X = filtered_ec2[feature_cols].fillna(0)
#             y = filtered_ec2[cost_col].fillna(0)

#             # Split data
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#             # Train model
#             model = LinearRegression()
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)

#             # --- Model Performance ---
#             st.subheader("📊 Model Performance")
#             st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")
#             st.write(f"**Mean Absolute Error (USD/hour):** {mean_absolute_error(y_test, y_pred):.4f}")

#             # --- Actual vs Predicted Plot ---
#             st.subheader("📈 Actual vs Predicted Costs")
#             result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
#             fig_pred = px.scatter(
#                 result_df, x="Actual", y="Predicted", trendline="ols",
#                 title="Actual vs Predicted Hourly Costs (USD)", template="plotly_white"
#             )
#             st.plotly_chart(fig_pred, use_container_width=True)

#             # --- Feature Importance ---
#             st.subheader("🔍 Feature Importance")
#             importance_df = pd.DataFrame({
#                 "Feature": feature_cols,
#                 "Coefficient": model.coef_
#             }).sort_values("Coefficient", ascending=False)

#             fig_importance = px.bar(
#                 importance_df, x="Feature", y="Coefficient",
#                 title="Feature Impact on Hourly Cost", template="plotly_white"
#             )
#             st.plotly_chart(fig_importance, use_container_width=True)

#             # --- Dynamic Input for User Prediction ---
#             st.subheader("🔮 Predict Hourly Cost for Custom Input")
#             input_data = {}

#             for feature in feature_cols[:3]:  # Only first 3 features for simplicity
#                 col_min = float(X[feature].min())
#                 col_max = float(X[feature].max())
#                 col_mean = float(X[feature].mean())

#                 # Safe dynamic range
#                 lower_bound = max(0.0, col_min)
#                 upper_bound = max(col_max * 1.5, col_mean * 10, 1e6)

#                 val = st.number_input(
#                     f"Enter value for {feature}",
#                     min_value=lower_bound,
#                     max_value=upper_bound,
#                     value=min(col_mean, upper_bound / 2),
#                     step=max(1.0, (upper_bound - lower_bound) / 100),
#                     help=f"Suggested range: {lower_bound:.2f} – {upper_bound:.2f}"
#                 )
#                 input_data[feature] = val

#             if st.button("Predict Cost"):
#                 new_df = pd.DataFrame([input_data])
#                 prediction = model.predict(new_df)[0]
#                 st.success(f"💰 Predicted Hourly Cost: **${prediction:.4f} USD/hour**")

#                 # --- Optional FinOps insight ---
#                 st.info(
#                     f"💡 If you reduced CPU utilization by 20%, the estimated cost could drop to "
#                     f"**${prediction * 0.8:.4f} USD/hour** (approximate savings)."
#                 )

#     else:
#         st.warning("⚠️ Required columns ('CPUUtilization' and 'CostPerHourUSD' or similar) not found in EC2 dataset.")



# =====================================================
# 🤖 TAB 4: Machine Learning Cost Optimization Model
# =====================================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

with tab4:
    st.header("🤖 Machine Learning Model – EC2 Cost Prediction")
    st.markdown("""
    This section uses a simple **Linear Regression** model to predict the **hourly cost (USD)** 
    of EC2 instances based on performance metrics such as CPU utilization, memory, or storage.
    It helps identify which metrics most influence cost, and estimate cost for new configurations.
    """)

    # --- Helper function to detect relevant columns ---
    def find_column(df, keywords):
        for col in df.columns:
            if any(k.lower() in col.lower().replace(" ", "") for k in keywords):
                return col
        return None

    # --- Automatically detect the key columns ---
    cpu_col = find_column(filtered_ec2, ["cpuutil", "cpuusage", "cpuutilization"])
    cost_col = find_column(filtered_ec2, ["costperhour", "hourlycost", "costusd", "priceusd"])

    # --- Display detected columns ---
    st.write("📋 Columns detected in EC2 dataset:", filtered_ec2.columns.tolist())

    if cpu_col and cost_col:
        st.success(f"✅ Detected columns: CPU → **{cpu_col}**, Cost → **{cost_col}**")

        # --- Select numeric columns for model training ---
        numeric_cols = filtered_ec2.select_dtypes(include=["float64", "int64"]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != cost_col]

        if len(feature_cols) == 0:
            st.warning("⚠️ No numeric columns found to train the model.")
        else:
            # Prepare training data
            X = filtered_ec2[feature_cols].fillna(0)
            y = filtered_ec2[cost_col].fillna(0)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # --- Model Performance ---
            st.subheader("📊 Model Performance")
            st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")
            st.write(f"**Mean Absolute Error (USD/hour):** {mean_absolute_error(y_test, y_pred):.4f}")

            # --- Actual vs Predicted Plot (with regression line) ---
            st.subheader("📈 Actual vs Predicted Costs")
            result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

            try:
                import statsmodels.api  # ensures availability for OLS trendline
                fig_pred = px.scatter(
                    result_df,
                    x="Actual",
                    y="Predicted",
                    trendline="ols",
                    title="Actual vs Predicted Hourly Costs (USD)",
                    template="plotly_white"
                )
            except ModuleNotFoundError:
                fig_pred = px.scatter(
                    result_df,
                    x="Actual",
                    y="Predicted",
                    title="Actual vs Predicted Hourly Costs (USD)",
                    template="plotly_white"
                )
                st.warning("⚠️ 'statsmodels' not installed — regression line hidden. Add it to requirements.txt to show it.")

            st.plotly_chart(fig_pred, use_container_width=True)

            # --- Feature Importance ---
            st.subheader("🔍 Feature Importance")
            importance_df = pd.DataFrame({
                "Feature": feature_cols,
                "Coefficient": model.coef_
            }).sort_values("Coefficient", ascending=False)

            fig_importance = px.bar(
                importance_df,
                x="Feature",
                y="Coefficient",
                title="Feature Impact on Hourly Cost",
                template="plotly_white"
            )
            st.plotly_chart(fig_importance, use_container_width=True)

            # --- Dynamic Input for User Prediction ---
            st.subheader("🔮 Predict Hourly Cost for Custom Input")
            input_data = {}

            for feature in feature_cols[:3]:  # Only first 3 features for simplicity
                col_min = float(X[feature].min())
                col_max = float(X[feature].max())
                col_mean = float(X[feature].mean())

                lower_bound = max(0.0, col_min)
                upper_bound = max(col_max * 1.5, col_mean * 10, 1e6)

                val = st.number_input(
                    f"Enter value for {feature}",
                    min_value=lower_bound,
                    max_value=upper_bound,
                    value=min(col_mean, upper_bound / 2),
                    step=max(1.0, (upper_bound - lower_bound) / 100),
                    help=f"Suggested range: {lower_bound:.2f} – {upper_bound:.2f}"
                )
                input_data[feature] = val

            if st.button("Predict Cost"):
                # 🩹 FIX: Ensure all training columns exist in new_df
                new_df = pd.DataFrame([input_data])
                for col in feature_cols:
                    if col not in new_df.columns:
                        new_df[col] = X[col].mean()  # fill missing with mean
                new_df = new_df[feature_cols]  # keep same column order

                prediction = model.predict(new_df)[0]
                st.success(f"💰 Predicted Hourly Cost: **${prediction:.4f} USD/hour**")

                st.info(
                    f"💡 If you reduced CPU utilization by 20%, the estimated cost could drop to "
                    f"**${prediction * 0.8:.4f} USD/hour** (approximate savings)."
                )

    else:
        st.warning("⚠️ Required columns ('CPUUtilization' and 'CostPerHourUSD' or similar) not found in EC2 dataset.")






# =====================================================
# Downloads
# =====================================================
st.download_button(
    label="📥 Download EC2 Filtered Data",
    data=filtered_ec2.to_csv(index=False).encode("utf-8"),
    file_name="Filtered_EC2_Data.csv",
    mime="text/csv"
)

st.download_button(
    label="📥 Download S3 Filtered Data",
    data=filtered_s3.to_csv(index=False).encode("utf-8"),
    file_name="Filtered_S3_Data.csv",
    mime="text/csv"
)

st.success("✅ EDA complete! Use the sidebar to adjust filters and upload new data.")
