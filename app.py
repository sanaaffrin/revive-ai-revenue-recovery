import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

st.set_page_config(
    page_title="REVIVE AI",
    page_icon="💰",
    layout="wide"
)

st.title("💰 REVIVE AI")
st.subheader("The Revenue Recovery Decision Engine")
st.write(
    "Find lost revenue. Predict what can be recovered. "
    "Take the right action."
)

# -----------------------------
# Generate demo customer data
# -----------------------------
np.random.seed(42)

n = 1200

df = pd.DataFrame({
    "customer_id": [f"CUST-{i:04d}" for i in range(1, n + 1)],
    "customer_value": np.random.randint(500, 25000, n),
    "days_since_purchase": np.random.randint(1, 180, n),
    "sessions_30d": np.random.randint(0, 30, n),
    "cart_value": np.random.randint(0, 12000, n),
    "payment_failures": np.random.randint(0, 4, n),
    "support_tickets": np.random.randint(0, 6, n),
    "discount_dependency": np.random.uniform(0, 1, n),
    "subscription": np.random.choice([0, 1], n),
    "last_payment_success": np.random.choice([0, 1], n, p=[0.18, 0.82])
})

df["revenue_at_risk"] = (
    df["cart_value"] * 0.45
    + df["customer_value"] * 0.08
    + df["payment_failures"] * 700
)

risk_signal = (
    df["payment_failures"] * 1.2
    + (df["days_since_purchase"] > 60) * 0.8
    + (df["sessions_30d"] < 5) * 0.7
    + df["discount_dependency"] * 0.6
    + (df["last_payment_success"] == 0) * 1.5
)

prob = 1 / (1 + np.exp(-(risk_signal - 2.4)))
df["recovery_label"] = np.random.binomial(1, prob)

# -----------------------------
# Train AI model
# -----------------------------
features = [
    "customer_value",
    "days_since_purchase",
    "sessions_30d",
    "cart_value",
    "payment_failures",
    "support_tickets",
    "discount_dependency",
    "subscription",
    "last_payment_success"
]

X = df[features]
y = df["recovery_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=180,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

test_probability = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, test_probability)

df["recovery_probability"] = model.predict_proba(X)[:, 1]
df["rescue_score"] = (df["recovery_probability"] * 100).round(0).astype(int)

# -----------------------------
# AI recommendation engine
# -----------------------------
def recommendation(row):
    reasons = []

    if row["payment_failures"] > 0:
        reasons.append("recent payment failure")

    if row["days_since_purchase"] > 60:
        reasons.append("customer inactivity")

    if row["cart_value"] > 3000:
        reasons.append("high-value cart")

    if row["last_payment_success"] == 0:
        reasons.append("latest payment unsuccessful")

    if row["sessions_30d"] < 5:
        reasons.append("low recent engagement")

    if row["recovery_probability"] >= 0.75:
        action = "Recover now"
    elif row["recovery_probability"] >= 0.50:
        action = "Monitor & engage"
    else:
        action = "Do not spend recovery budget"

    reason = ", ".join(reasons[:3])

    return pd.Series([action, reason])


df[["recommended_action", "reason"]] = df.apply(
    recommendation,
    axis=1
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Recovery Controls")

min_probability = st.sidebar.slider(
    "Minimum recovery probability",
    0.0,
    1.0,
    0.60,
    0.05
)

min_revenue = st.sidebar.slider(
    "Minimum revenue at risk",
    0,
    10000,
    1000,
    500
)

queue = df[
    (df["recovery_probability"] >= min_probability)
    & (df["revenue_at_risk"] >= min_revenue)
].copy()

# -----------------------------
# Dashboard metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Revenue at Risk",
        f"₹{df['revenue_at_risk'].sum():,.0f}"
    )

with col2:
    st.metric(
        "High-Probability Opportunities",
        len(queue)
    )

with col3:
    expected_recovery = (
        queue["revenue_at_risk"]
        * queue["recovery_probability"]
    ).sum()

    st.metric(
        "Expected Recoverable Revenue",
        f"₹{expected_recovery:,.0f}"
    )

with col4:
    st.metric(
        "Model Validation AUC",
        f"{auc:.2f}"
    )

st.divider()

# -----------------------------
# Recovery queue
# -----------------------------
st.header("🚨 AI Recovery Queue")

display_columns = [
    "customer_id",
    "revenue_at_risk",
    "recovery_probability",
    "rescue_score",
    "recommended_action",
    "reason"
]

display_df = queue[display_columns].sort_values(
    "rescue_score",
    ascending=False
).head(25)

display_df["revenue_at_risk"] = (
    display_df["revenue_at_risk"].round(0)
)

display_df["recovery_probability"] = (
    display_df["recovery_probability"] * 100
).round(1).astype(str) + "%"

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Charts
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue Leakage Signals")

    leakage = pd.DataFrame({
        "Signal": [
            "Payment Failures",
            "Customer Inactivity",
            "Low Engagement",
            "High Cart Value"
        ],
        "Customers": [
            int((df["payment_failures"] > 0).sum()),
            int((df["days_since_purchase"] > 60).sum()),
            int((df["sessions_30d"] < 5).sum()),
            int((df["cart_value"] > 3000).sum())
        ]
    })

    fig = px.bar(
        leakage,
        x="Signal",
        y="Customers",
        title="Where Revenue Is Leaking"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Rescue Score Distribution")

    fig = px.histogram(
        df,
        x="rescue_score",
        nbins=20,
        title="AI Recovery Opportunity Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Customer explanation
# -----------------------------
st.divider()
st.header("🔎 Customer-Level AI Explanation")

customer = st.selectbox(
    "Select a customer",
    df["customer_id"].tolist()
)

row = df[df["customer_id"] == customer].iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Revenue at Risk",
        f"₹{row['revenue_at_risk']:,.0f}"
    )

with col2:
    st.metric(
        "Recovery Probability",
        f"{row['recovery_probability'] * 100:.1f}%"
    )

with col3:
    st.metric(
        "Rescue Score",
        f"{row['rescue_score']}/100"
    )

st.info(
    f"**AI Recommendation:** {row['recommended_action']}\n\n"
    f"**Why:** {row['reason'] or 'No major risk signal detected.'}"
)

st.caption(
    "Demo note: this version uses synthetic data and simulated "
    "recovery outcomes for demonstration purposes."
)

st.divider()

st.subheader("How REVIVE AI works")

st.write(
    "Customer data → AI risk analysis → Recovery probability → "
    "Rescue Score → Recommended action → Revenue recovery"
)
