import os
import pickle
import traceback
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model

st.set_page_config(page_title="AI 고양이 탐지기", page_icon="🐾", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Gowun+Dodum&display=swap');
html, body, [class*="css"] { font-family: 'Gowun Dodum', sans-serif; }
.result-cat {
    background: linear-gradient(135deg, #fff0f6, #ffe4f0);
    border-left: 5px solid #ff69b4;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-size: 1.4rem;
    font-weight: bold;
    color: #c2185b;
    margin-top: 1rem;
}
.result-nocat {
    background: linear-gradient(135deg, #f0f4ff, #e8eeff);
    border-left: 5px solid #5c6bc0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-size: 1.4rem;
    font-weight: bold;
    color: #3949ab;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="AI 모델 불러오는 중...")
def load_embedder():
    base = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    return Model(inputs=base.input, outputs=base.output)

@st.cache_resource(show_spinner="분류기 불러오는 중...")
def load_classifier():
    model_path = "cat_detector_model.pkl"
    if not os.path.exists(model_path):
        return None, f"`{model_path}` 파일을 찾을 수 없습니다."
    try:
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        return clf, None
    except Exception as e:
        return None, f"모델 로드 실패: {e}"

def embed_image(pil_image):
    img = pil_image.convert("RGB").resize((299, 299), Image.LANCZOS)
    arr = preprocess_input(np.array(img, dtype=np.float32)[np.newaxis, ...])
    result = embedder_model(arr, training=False)
    return result.numpy()[0]

def predict(features, clf):
    import Orange.data as od
    domain = clf.domain

    n_attrs = len(domain.attributes)
    padded = np.zeros((1, n_attrs))
    padded[0, :len(features)] = features

    n_class = len(domain.class_vars)
    Y = np.full((1, n_class), np.nan)

    n_metas = len(domain.metas)
    metas = np.full((1, n_metas), np.nan, dtype=object)

    table = od.Table.from_numpy(domain, padded, Y, metas)
    probs = np.array(clf(table, clf.Probs)[0])
    label_idx = int(np.argmax(probs))
    label = str(domain.class_var.values[label_idx])
    confidence = float(probs[label_idx])
    all_probs = list(zip([str(v) for v in domain.class_var.values], probs.tolist()))
    return label, confidence, all_probs

st.title("ฅ^. .^= 올인원 고양이 탐지기 🐾")
st.caption("밈, 그림, 흐릿한 사진에서도 고양이를 인식하도록 학습된 AI")

embedder_model = load_embedder()
classifier, err = load_classifier()

if err:
    st.error(err)
    st.stop()

st.success("모델 준비 완료! 이미지를 업로드해보세요.")
st.divider()

uploaded_file = st.file_uploader("이미지 선택 (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="업로드된 이미지", width=700)
    st.write("₍^. .^₎⟆ 이미지 분석 중...")

    with st.spinner(""):
        try:
            features = embed_image(img)
            label, confidence, all_probs = predict(features, classifier)

            if label == "고양이":
                st.markdown(
                    f'<div class="result-cat">ฅ^._.^ฅ 고양이입니다! &nbsp; ({confidence*100:.1f}% 확신)</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-nocat">𑄝 고양이가 아닙니다 &nbsp; ({confidence*100:.1f}% 확신)</div>',
                    unsafe_allow_html=True
                )

            st.markdown(f"**확신도:** {confidence*100:.1f}%")
            st.progress(confidence)

            with st.expander("전체 클래스 확률 보기"):
                for cls_name, prob in sorted(all_probs, key=lambda x: -x[1]):
                    st.write(f"**{cls_name}**: {prob*100:.1f}%")
                    st.progress(prob)

        except Exception as e:
            st.error(f"⚠️ 오류 발생: {e}")
            st.code(traceback.format_exc(), language="python")

st.divider()
st.caption("2555041 / ESCOBEDO JIMENEZ ANGIE ALEJANDRA")