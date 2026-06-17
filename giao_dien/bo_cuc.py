import streamlit as st


def chen_css_co_ban():
    # Chèn CSS nhẹ để bố cục gọn gàng hơn.
    st.markdown(
        """
<style>
.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 18px;
    border: 1px solid rgba(90, 90, 90, 0.14);
    background: linear-gradient(120deg, rgba(243, 248, 255, 0.95), rgba(255, 255, 255, 0.85));
    margin-bottom: 1.6rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1rem;
    margin-top: 0.3rem;
    color: rgba(80, 80, 80, 0.85);
}
.section-title {
    font-size: 1.35rem;
    font-weight: 600;
    margin: 1rem 0 0.3rem 0;
}
.card {
    padding: 1rem 1.2rem;
    border-radius: 14px;
    border: 1px solid rgba(90, 90, 90, 0.12);
    background: rgba(255, 255, 255, 0.75);
}
@media (prefers-color-scheme: dark) {
    .hero {
        background: linear-gradient(120deg, rgba(20, 24, 30, 0.92), rgba(40, 40, 40, 0.7));
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-subtitle {
        color: rgba(220, 220, 220, 0.8);
    }
    .card {
        background: rgba(20, 20, 20, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def hien_tieu_de_dau_trang():
    """Hiển thị khu vực tiêu đề đầu trang."""
    st.markdown(
        """
<div class="hero">
  <div class="hero-title">Used Car Price AI Demo</div>
  <div class="hero-subtitle">
    Tabular pricing, damage detection, severity grading, and price adjustment in one demo.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def hien_tieu_de_muc(tieu_de, phu_de=None):
    """Hiển thị tiêu đề cho từng mục nội dung."""
    st.markdown(f"<div class='section-title'>{tieu_de}</div>", unsafe_allow_html=True)
    if phu_de:
        st.caption(phu_de)
