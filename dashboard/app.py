from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "outputs" / "cross_section.csv"


@st.cache_data
def load_csv(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    return df


def main() -> None:
    st.set_page_config(page_title="Cross-section factors", layout="wide")
    st.title("Cross-section: CAPM vs Fama–French")
    st.caption("Load `outputs/cross_section.csv` or upload your own file.")

    path_input = st.sidebar.text_input(
        "CSV path (relative to repo or absolute)",
        value=str(DEFAULT_CSV),
    )
    uploaded = st.sidebar.file_uploader("Or upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "error" in df.columns:
            df = df[df["error"].isna()].copy()
    else:
        df = load_csv(path_input)

    if df.empty:
        st.warning(
            "No data. Run `python utils/fama_french.py --step3` from the repo root, "
            "or fix the path / upload a CSV."
        )
        return

    tickers = sorted(df["ticker"].dropna().unique().tolist()) if "ticker" in df.columns else []
    pick = st.sidebar.multiselect("Tickers", tickers, default=tickers)
    if pick:
        df = df[df["ticker"].isin(pick)]

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    c1, c2, c3, c4 = st.columns(4)
    if "ff3_r2" in df.columns:
        c1.metric("Mean FF3 R²", f"{df['ff3_r2'].mean():.4f}")
    if "capm_r2" in df.columns:
        c2.metric("Mean CAPM R²", f"{df['capm_r2'].mean():.4f}")
    if "ff3_mkt" in df.columns:
        c3.metric("Mean FF3 MKT loading", f"{df['ff3_mkt'].mean():.4f}")
    if "capm_beta_mkt" in df.columns:
        c4.metric("Mean CAPM β", f"{df['capm_beta_mkt'].mean():.4f}")

    tab1, tab2, tab3 = st.tabs(["Explore", "Scatter matrix", "Table"])

    with tab1:
        xa = st.selectbox("X axis", num_cols, index=num_cols.index("capm_beta_mkt") if "capm_beta_mkt" in num_cols else 0)
        ya = st.selectbox("Y axis", num_cols, index=num_cols.index("ff3_mkt") if "ff3_mkt" in num_cols else min(1, len(num_cols) - 1))
        color = st.selectbox("Color (optional)", [None] + num_cols, index=0)
        fig = px.scatter(
            df,
            x=xa,
            y=ya,
            color=color if color else None,
            hover_name="ticker" if "ticker" in df.columns else None,
            trendline="ols" if len(df) >= 3 else None,
        )
        st.plotly_chart(fig, use_container_width=True)

        if {"capm_r2", "ff3_r2"}.issubset(df.columns):
            fig2 = px.scatter(
                df,
                x="capm_r2",
                y="ff3_r2",
                hover_name="ticker" if "ticker" in df.columns else None,
                labels={"capm_r2": "CAPM R²", "ff3_r2": "FF3 R²"},
                title="CAPM R² vs FF3 R²",
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        cols_matrix = [c for c in ["capm_beta_mkt", "ff3_mkt", "ff3_smb", "ff3_hml", "capm_r2", "ff3_r2"] if c in df.columns]
        if len(cols_matrix) >= 2:
            figm = px.scatter_matrix(df, dimensions=cols_matrix, hover_name="ticker" if "ticker" in df.columns else None)
            figm.update_traces(diagonal_visible=False)
            st.plotly_chart(figm, use_container_width=True)
        else:
            st.info("Not enough numeric columns for scatter matrix.")

    with tab3:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="cross_section_filtered.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
