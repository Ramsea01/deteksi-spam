import streamlit as st
import joblib
import pandas as pd


@st.cache_resource
def load_models():
    vectorizer = joblib.load("tfidf.pkl")
    rf = joblib.load("rf_model.pkl")
    return vectorizer, rf


vectorizer, rf = load_models()

st.set_page_config(page_title="Deteksi Email Spam", page_icon="📧")
st.title("📧 Deteksi Email Spam")
st.write("Masukkan subjek dan isi email, lalu klik **Periksa**.")

# Sidebar: pengaturan threshold
st.sidebar.header("Pengaturan")
threshold = st.sidebar.slider(
    "Threshold spam", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="Email dianggap spam jika probabilitas spam >= nilai ini."
)

subject = st.text_input("Subjek email")
body = st.text_area("Isi email", height=200)

if st.button("Periksa", type="primary"):
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

        # Hasil utama
        if spam_pct / 100 >= threshold:
            st.error(f"🚨 Terdeteksi **SPAM**")
        else:
            st.success(f"✅ Terdeteksi **BUKAN SPAM (HAM)**")

        # Persentase
        col1, col2 = st.columns(2)
        col1.metric("Ham (bukan spam)", f"{ham_pct:.1f}%")
        col2.metric("Spam", f"{spam_pct:.1f}%")

        # Grafik batang
        chart_df = pd.DataFrame(
            {"Persentase (%)": [ham_pct, spam_pct]},
            index=["Ham", "Spam"],
        )
        st.bar_chart(chart_df)

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
                df = pd.DataFrame(data, columns=["Kata", "Bobot TF-IDF", "Importance model"])
                df = df.sort_values("Importance model", ascending=False).head(10)
                st.dataframe(df, hide_index=True)