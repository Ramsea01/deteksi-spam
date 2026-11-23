import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Deteksi Email Spam", page_icon="📧", layout="centered")

# ---------- Palet warna earth tone ----------
CREAM = "#F5EFE6"
SAND = "#E6D5B8"
BROWN = "#5C4033"
DARK_BROWN = "#3E2C1E"
TERRACOTTA = "#B5573A"
OLIVE = "#6B7A3A"
CLAY = "#C98B5E"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, {CREAM} 0%, {SAND} 100%);
    }}
    .stApp, .stApp p, .stApp label, .stApp span, .stApp li {{
        color: {DARK_BROWN};
    }}
    h1, h2, h3 {{
        color: {BROWN} !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BROWN} 0%, {DARK_BROWN} 100%);
    }}
    [data-testid="stSidebar"] * {{
        color: {CREAM} !important;
    }}
    .member {{
        background: rgba(245, 239, 230, 0.10);
        border-left: 4px solid {CLAY};
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 8px;
    }}
    .member .name {{ font-weight: 600; font-size: 0.95rem; }}
    .member .nim {{ font-size: 0.8rem; opacity: 0.85; }}

    /* Input */
    .stTextInput input, .stTextArea textarea {{
        background-color: #FFFAF2 !important;
        color: {DARK_BROWN} !important;
        border: 1.5px solid {CLAY} !important;
        border-radius: 10px !important;
    }}

    /* Tombol */
    .stButton > button {{
        background-color: {TERRACOTTA};
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
    }}
    .stButton > button p, .stButton > button span {{
        color: #FFFFFF !important;
    }}
    .stButton > button:hover {{
        background-color: {BROWN};
        color: #FFFFFF;
    }}

    /* Catatan bahasa */
    .note {{
        background-color: rgba(201, 139, 94, 0.18);
        border-left: 5px solid {CLAY};
        border-radius: 10px;
        padding: 12px 16px;
        margin: 14px 0 6px 0;
        font-size: 0.9rem;
        color: {DARK_BROWN};
    }}

    /* Banner judul */
    .hero {{
        background: linear-gradient(135deg, {BROWN} 0%, {TERRACOTTA} 100%);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
    }}
    .hero h1 {{ color: {CREAM} !important; margin: 0; font-size: 2rem; }}
    .hero p {{ color: {SAND} !important; margin: 6px 0 0 0; }}

    /* Kartu hasil */
    .result {{
        border-radius: 14px;
        padding: 18px 22px;
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin: 10px 0 18px 0;
    }}
    .result.spam {{ background-color: {TERRACOTTA}; }}
    .result.ham {{ background-color: {OLIVE}; }}

    /* Kartu persentase */
    .card {{
        background-color: #FFFAF2;
        border: 1.5px solid {CLAY};
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }}
    .card .label {{ font-size: 0.9rem; color: {BROWN}; }}
    .card .value {{ font-size: 2rem; font-weight: 700; }}
    .bar-bg {{
        background-color: {SAND};
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
        margin-top: 6px;
    }}
    .bar-fill {{ height: 100%; border-radius: 8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    vectorizer = joblib.load("tfidf.pkl")
    rf = joblib.load("rf_model.pkl")
    return vectorizer, rf


vectorizer, rf = load_models()

# ---------- Sidebar: info kelompok ----------
anggota = [
    ("Kaka Pratama Lepar", "2310631160064"),
    ("Faisal Dzikri Ramdhani", "2310631160056"),
    ("Muhammad Thoriq Syah Gumay", "2310631160072"),
    ("Illona Elvaretta Adzhani", "2310631160017"),
    ("Nabilla Alya Septiani", "2310631160027"),
]

st.sidebar.markdown("## 👥 Kelompok 3")
st.sidebar.markdown("**Anggota:**")
for nama, nim in anggota:
    st.sidebar.markdown(
        f'<div class="member"><div class="name">{nama}</div>'
        f'<div class="nim">{nim}</div></div>',
        unsafe_allow_html=True,
    )

# ---------- Halaman utama ----------
st.markdown(
    """
    <div class="hero">
        <h1>📧 Deteksi Email Spam</h1>
        <p>Masukkan subjek dan isi email, lalu klik <b>Periksa</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

subject = st.text_input("Subjek email")
body = st.text_area("Isi email", height=200)

clicked = st.button("Periksa", type="primary")

st.markdown(
    """
    <div class="note">
        📝 <b>Catatan:</b> Model ini dilatih menggunakan data email berbahasa Inggris,
        jadi hasil deteksi paling optimal jika subjek dan isi email ditulis dalam
        <b>bahasa Inggris</b>. Email berbahasa lain mungkin menghasilkan prediksi yang kurang akurat.
    </div>
    """,
    unsafe_allow_html=True,
)

if clicked:
    if subject.strip() == "" and body.strip() == "":
        st.warning("Subjek dan isi email tidak boleh kosong.")
    else:
        # Gabungkan subjek dan isi menjadi satu teks
        text = f"{subject} {body}".strip()

        # Tambahkan preprocessing di sini jika di Colab kamu juga melakukannya
        # (huruf kecil, hapus tanda baca, stemming, dll.)

        X = vectorizer.transform([text])
        proba = rf.predict_proba(X)[0]

        # Cari indeks kelas berdasarkan label, jangan asumsi urutan
        classes = list(rf.classes_)
        # Sesuaikan jika labelmu berupa string, misalnya "spam" / "ham"
        spam_label = 1 if 1 in classes else "spam"
        ham_label = 0 if 0 in classes else "ham"

        spam_pct = proba[classes.index(spam_label)] * 100
        ham_pct = proba[classes.index(ham_label)] * 100

        # Hasil utama (batas tetap 50%)
        if spam_pct >= 50:
            st.markdown(
                '<div class="result spam">🚨 Terdeteksi SPAM</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result ham">✅ Terdeteksi BUKAN SPAM (HAM)</div>',
                unsafe_allow_html=True,
            )

        # Persentase
        col1, col2 = st.columns(2)
        col1.markdown(
            f"""
            <div class="card">
                <div class="label">Ham (bukan spam)</div>
                <div class="value" style="color:{OLIVE}">{ham_pct:.1f}%</div>
                <div class="bar-bg"><div class="bar-fill" style="width:{ham_pct:.1f}%; background:{OLIVE}"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col2.markdown(
            f"""
            <div class="card">
                <div class="label">Spam</div>
                <div class="value" style="color:{TERRACOTTA}">{spam_pct:.1f}%</div>
                <div class="bar-bg"><div class="bar-fill" style="width:{spam_pct:.1f}%; background:{TERRACOTTA}"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Kata paling berpengaruh pada email ini
        with st.expander("Kata yang paling berpengaruh pada email ini"):
            feature_names = vectorizer.get_feature_names_out()
            row = X.tocoo()
            if row.nnz == 0:
                st.write("Tidak ada kata dalam email yang dikenali oleh model.")
            else:
                importances = rf.feature_importances_
                data = [
                    (feature_names[j], v, importances[j])
                    for j, v in zip(row.col, row.data)
                ]
                df = pd.DataFrame(
                    data, columns=["Kata", "Bobot TF-IDF", "Importance model"]
                )
                df = df.sort_values("Importance model", ascending=False).head(10)
                st.dataframe(df, hide_index=True)