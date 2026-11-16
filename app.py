import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="SMARTSHELF AI", page_icon="🛍️", layout="wide")


# Mobile-friendly layout for phones and tablets
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    padding-left: clamp(0.7rem, 3vw, 2.5rem);
    padding-right: clamp(0.7rem, 3vw, 2.5rem);
    max-width: 1500px;
}
h1 { font-size: clamp(1.65rem, 5vw, 2.5rem) !important; }
h2, h3 { font-size: clamp(1.15rem, 3.8vw, 1.65rem) !important; }
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.18);
    padding: 0.7rem;
    border-radius: 0.75rem;
}
div[data-testid="stDataFrame"] { width: 100%; }
@media (max-width: 700px) {
    .block-container {
        padding-left: 0.65rem;
        padding-right: 0.65rem;
    }
    div[data-testid="stMetric"] { padding: 0.5rem; }
    div[data-testid="stMetricLabel"] p { font-size: 0.75rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.25rem !important; }
    button[data-baseweb="tab"] { padding-left: 0.45rem; padding-right: 0.45rem; }
    [data-testid="stHorizontalBlock"] { gap: 0.5rem; }
}
</style>
""", unsafe_allow_html=True)


st.title("🛍️ SMARTSHELF AI")
st.caption("Fashion Retail Intelligence • Demo prototype for clothing stores")

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv("data/sample_sales.csv")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload sales CSV", type=["csv"])
    st.divider()
    st.header("Replenishment settings")
    lead_time = st.number_input("Supplier lead time (days)", min_value=1, max_value=60, value=7)
    target_cover = st.number_input("Target stock after delivery (days)", min_value=1, max_value=90, value=14)

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Could not load the CSV: {e}")
    st.stop()

required = {"date", "product", "category", "size", "color", "units_sold", "stock", "price", "cost"}
missing = required - set(df.columns)
if missing:
    st.error("CSV is missing required columns: " + ", ".join(sorted(missing)))
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
for col in ["units_sold", "stock", "price", "cost"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
df = df.dropna(subset=["date", "product"])

df["revenue"] = df["units_sold"] * df["price"]
df["gross_profit"] = df["units_sold"] * (df["price"] - df["cost"])

total_revenue = df["revenue"].sum()
units = df["units_sold"].sum()
profit = df["gross_profit"].sum()
days = max(df["date"].dt.date.nunique(), 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue", f"{total_revenue:,.0f}")
c2.metric("Units sold", f"{units:,.0f}")
c3.metric("Estimated gross profit", f"{profit:,.0f}")
c4.metric("Products", f"{df['product'].nunique()}")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Demand Forecast", "Size Intelligence",
    "Color Intelligence", "Smart Replenishment", "Business Advisor",
    "Profit Intelligence"
])

with tab1:
    st.subheader("Sales overview")
    daily = df.groupby(df["date"].dt.date, as_index=False)["revenue"].sum()
    daily.columns = ["date", "revenue"]
    st.plotly_chart(px.line(daily, x="date", y="revenue", markers=True, title="Daily revenue"), use_container_width=True)
    category = df.groupby("category", as_index=False)["units_sold"].sum()
    st.plotly_chart(px.bar(category, x="category", y="units_sold", title="Units sold by category"), use_container_width=True)

with tab2:
    st.subheader("Demand Forecast")
    st.info("Baseline estimate using average observed daily sales. It is not a trained machine-learning model.")
    forecast_days = st.selectbox("Forecast horizon", [7, 14, 30], index=0)
    avg_daily = units / days
    forecast_units = avg_daily * forecast_days
    avg_price = (total_revenue / units) if units else 0
    a, b = st.columns(2)
    a.metric(f"Estimated units for next {forecast_days} days", f"{forecast_units:,.1f}")
    b.metric("Estimated revenue", f"{forecast_units * avg_price:,.0f}")
    st.caption(f"Calculation: {units:,.0f} units across {days} observed sales days.")

with tab3:
    st.subheader("Size Intelligence")
    size_data = df.groupby("size", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False)
    st.plotly_chart(px.bar(size_data, x="size", y="units_sold", title="Units sold by size"), use_container_width=True)
    st.dataframe(size_data, use_container_width=True)

with tab4:
    st.subheader("Color Intelligence")
    color_data = df.groupby("color", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False)
    st.plotly_chart(px.bar(color_data, x="color", y="units_sold", title="Units sold by color"), use_container_width=True)
    st.dataframe(color_data, use_container_width=True)

with tab5:
    st.subheader("Smart Replenishment")
    st.write("Estimate which product variants may need replenishment based on observed sales velocity and current stock.")
    latest = df.sort_values("date").groupby(["product", "category", "size", "color"], as_index=False).tail(1)
    sold = df.groupby(["product", "category", "size", "color"], as_index=False).agg(
        total_units_sold=("units_sold", "sum"),
        avg_price=("price", "last"),
        avg_cost=("cost", "last"),
        active_days=("date", lambda x: max(x.dt.date.nunique(), 1))
    )
    stock = latest[["product", "category", "size", "color", "stock"]]
    repl = sold.merge(stock, on=["product", "category", "size", "color"], how="left")
    repl["avg_daily_sales"] = repl["total_units_sold"] / repl["active_days"]
    repl["days_of_stock"] = np.where(repl["avg_daily_sales"] > 0, repl["stock"] / repl["avg_daily_sales"], np.inf)
    repl["target_stock"] = np.ceil(repl["avg_daily_sales"] * (lead_time + target_cover))
    repl["recommended_order"] = np.maximum(repl["target_stock"] - repl["stock"], 0).astype(int)

    def status(row):
        if row["total_units_sold"] <= 0:
            return "No sales data"
        if row["days_of_stock"] <= lead_time:
            return "Urgent"
        if row["recommended_order"] > 0:
            return "Reorder suggested"
        return "Stock sufficient"

    repl["status"] = repl.apply(status, axis=1)
    urgent_count = int((repl["status"] == "Urgent").sum())
    order_count = int((repl["recommended_order"] > 0).sum())
    order_units = int(repl["recommended_order"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Variants to reorder", order_count)
    m2.metric("Suggested order units", order_units)
    m3.metric("Urgent variants", urgent_count)

    show = repl.copy()
    show["days_of_stock"] = show["days_of_stock"].replace(np.inf, np.nan).round(1)
    show = show.rename(columns={
        "product": "Product", "category": "Category", "size": "Size", "color": "Color",
        "total_units_sold": "Units sold", "stock": "Current stock",
        "avg_daily_sales": "Avg daily sales", "days_of_stock": "Days of stock",
        "target_stock": "Target stock", "recommended_order": "Recommended order",
        "status": "Status"
    })
    st.dataframe(show[["Product", "Category", "Size", "Color", "Units sold", "Current stock",
                       "Avg daily sales", "Days of stock", "Target stock",
                       "Recommended order", "Status"]], use_container_width=True)
    st.download_button(
        "Download replenishment plan (CSV)",
        data=show.to_csv(index=False).encode("utf-8-sig"),
        file_name="smartshelf_replenishment_plan.csv",
        mime="text/csv"
    )
    st.warning("Demo limitation: recommendations are estimates from the sample sales history and the latest recorded stock. Validate with real inventory, supplier schedules and store context before ordering.")


with tab6:
    st.subheader("AI Business Advisor")
    st.caption("Decision-support recommendations generated from the uploaded sales and stock data.")

    category_sales = df.groupby("category", as_index=False).agg(
        units=("units_sold", "sum"), revenue=("revenue", "sum")
    ).sort_values("units", ascending=False)
    size_sales = df.groupby("size", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False)
    color_sales = df.groupby("color", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False)
    product_sales = df.groupby(["product", "category"], as_index=False).agg(
        units=("units_sold", "sum"), revenue=("revenue", "sum")
    ).sort_values("units", ascending=False)

    latest_stock = df.sort_values("date").groupby(
        ["product", "category", "size", "color"], as_index=False
    ).tail(1)
    variant_sales = df.groupby(["product", "category", "size", "color"], as_index=False).agg(
        units=("units_sold", "sum"),
        observed_days=("date", lambda x: max(x.dt.date.nunique(), 1))
    )
    inventory = variant_sales.merge(
        latest_stock[["product", "category", "size", "color", "stock"]],
        on=["product", "category", "size", "color"], how="left"
    )
    inventory["daily_rate"] = inventory["units"] / inventory["observed_days"]
    inventory["stock_days"] = np.where(
        inventory["daily_rate"] > 0, inventory["stock"] / inventory["daily_rate"], np.inf
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Categories analysed", int(df["category"].nunique()))
    k2.metric("Product variants", int(len(inventory)))
    k3.metric("Observed sales days", int(days))

    st.markdown("### Business recommendations")
    recommendations = []
    if len(category_sales):
        r = category_sales.iloc[0]
        recommendations.append(
            f"**Prioritise {r['category']} for attention.** It leads this dataset with {r['units']:,.0f} units sold and revenue of {r['revenue']:,.0f}."
        )
    if len(size_sales):
        r = size_sales.iloc[0]
        recommendations.append(
            f"**Review size allocation.** Size {r['size']} has the highest recorded demand ({r['units_sold']:,.0f} units). Compare its stock availability with other sizes."
        )
    if len(color_sales):
        r = color_sales.iloc[0]
        recommendations.append(
            f"**Review colour allocation.** {r['color']} is the leading colour in this sample ({r['units_sold']:,.0f} units sold)."
        )
    low_cover = inventory[(inventory["daily_rate"] > 0) & (inventory["stock_days"] <= lead_time)]
    if len(low_cover):
        urgent = low_cover.sort_values("stock_days").head(3)
        names = ", ".join(f"{r.product} ({r.size}/{r.color})" for r in urgent.itertuples())
        recommendations.append(
            f"**Check low-stock variants first.** Estimated stock cover is at or below the {lead_time}-day supplier lead time for: {names}."
        )
    else:
        recommendations.append("**No urgent stock-cover signal was found** using the current lead-time setting. Continue monitoring sales and stock.")
    if len(product_sales) > 1:
        r = product_sales.sort_values("units").iloc[0]
        recommendations.append(
            f"**Review slower-moving product {r['product']}.** It has the lowest recorded unit sales ({r['units']:,.0f}) in this sample. Check pricing, visibility and seasonality before making changes."
        )
    for i, item in enumerate(recommendations, 1):
        st.markdown(f"{i}. {item}")

    st.markdown("### Explore the data")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(category_sales, x="category", y="units", title="Demand by category"), use_container_width=True)
        st.plotly_chart(px.bar(size_sales, x="size", y="units_sold", title="Demand by size"), use_container_width=True)
    with right:
        st.plotly_chart(px.bar(product_sales.head(8), x="product", y="units", title="Top products by units sold"), use_container_width=True)
        st.plotly_chart(px.bar(color_sales, x="color", y="units_sold", title="Demand by colour"), use_container_width=True)

    st.info(
        "Transparency note: this advisor uses transparent business rules and descriptive statistics, "
        "not a connected generative-AI model or a trained machine-learning model. The included sample data is synthetic. "
        "Treat recommendations as prompts for review, not automatic purchasing decisions."
    )


with tab7:
    st.subheader("Profit Intelligence")
    st.caption("Understand where sales revenue turns into gross profit using the uploaded data.")

    profit_df = df.copy()
    total_cost = (profit_df["units_sold"] * profit_df["cost"]).sum()
    total_profit = profit_df["gross_profit"].sum()
    margin_pct = total_profit / total_revenue * 100 if total_revenue else 0
    profit_per_unit = total_profit / units if units else 0

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Gross profit", f"{total_profit:,.0f}")
    p2.metric("Cost of goods sold", f"{total_cost:,.0f}")
    p3.metric("Gross margin", f"{margin_pct:.1f}%")
    p4.metric("Profit per unit", f"{profit_per_unit:,.0f}")

    st.markdown("### Profit by category")
    category_profit = profit_df.groupby("category", as_index=False).agg(
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        units=("units_sold", "sum")
    )
    category_cost = profit_df.assign(cogs=profit_df["units_sold"] * profit_df["cost"]).groupby(
        "category", as_index=False
    )["cogs"].sum()
    category_profit = category_profit.merge(category_cost, on="category", how="left")
    category_profit["gross_margin_pct"] = np.where(
        category_profit["revenue"] > 0,
        category_profit["gross_profit"] / category_profit["revenue"] * 100, 0
    )
    st.plotly_chart(px.bar(category_profit, x="category", y="gross_profit",
                           title="Gross profit by category"), use_container_width=True)
    st.dataframe(category_profit.rename(columns={
        "category": "Category", "revenue": "Revenue", "cogs": "Cost of goods sold",
        "gross_profit": "Gross profit", "gross_margin_pct": "Gross margin (%)",
        "units": "Units sold"
    }).round(2), use_container_width=True)

    st.markdown("### Product profitability")
    product_profit = profit_df.groupby(["product", "category"], as_index=False).agg(
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        units=("units_sold", "sum")
    )
    product_cost = profit_df.assign(cogs=profit_df["units_sold"] * profit_df["cost"]).groupby(
        ["product", "category"], as_index=False
    )["cogs"].sum()
    product_profit = product_profit.merge(product_cost, on=["product", "category"], how="left")
    product_profit["gross_margin_pct"] = np.where(
        product_profit["revenue"] > 0,
        product_profit["gross_profit"] / product_profit["revenue"] * 100, 0
    )
    product_profit = product_profit.sort_values("gross_profit", ascending=False)
    st.plotly_chart(px.bar(product_profit.head(10), x="product", y="gross_profit",
                           title="Top products by gross profit"), use_container_width=True)
    st.dataframe(product_profit.rename(columns={
        "product": "Product", "category": "Category", "revenue": "Revenue",
        "cogs": "Cost of goods sold", "gross_profit": "Gross profit",
        "gross_margin_pct": "Gross margin (%)", "units": "Units sold"
    }).round(2), use_container_width=True)

    st.download_button("Download profitability report (CSV)",
        data=product_profit.to_csv(index=False).encode("utf-8-sig"),
        file_name="smartshelf_profitability_report.csv", mime="text/csv")
    st.info("Gross profit is revenue minus recorded product cost. It excludes rent, salaries, taxes, delivery "
            "and other operating expenses. Results depend on accurate price and cost data.")

with st.expander("Data preview and definitions"):
    st.dataframe(df, use_container_width=True)
    st.markdown("""
    - **Revenue** = units sold × selling price.
    - **Estimated gross profit** = units sold × (selling price − unit cost).
    - **Average daily sales** = units sold ÷ number of observed sales dates.
    - **Days of stock** = current stock ÷ average daily sales.
    - **Recommended order** = target stock − current stock, with a minimum of zero.
    """)
