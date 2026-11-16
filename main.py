import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Music Listening Habits Analysis", page_icon="🎵", layout="wide")
st.title("🎵 Music Listening Habits Analysis")
st.write("음악 감상 습관 데이터를 분석하고 선호 음악 장르를 예측합니다.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "music_listening_dataset.csv")

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

try:
    df = load_data()
except FileNotFoundError:
    st.error("CSV 파일을 찾을 수 없습니다. main.py와 CSV 파일을 같은 폴더에 넣어 주세요.")
    st.stop()

required = ["Age", "Listening_Hours", "Preferred_Time", "Mood", "Device", "Favorite_Genre"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("CSV에 필요한 열이 없습니다: " + ", ".join(missing))
    st.stop()

st.header("1. Dataset Overview")
st.dataframe(df.head(10), use_container_width=True)
st.write(f"데이터 크기: {df.shape[0]}행 × {df.shape[1]}열")
c1, c2, c3 = st.columns(3)
c1.metric("전체 데이터 수", len(df))
c2.metric("음악 장르 수", df["Favorite_Genre"].nunique())
c3.metric("평균 감상 시간", f'{df["Listening_Hours"].mean():.1f}시간')

st.header("2. Music Listening Analysis")
left, right = st.columns(2)
with left:
    st.subheader("선호 음악 장르")
    fig, ax = plt.subplots()
    df["Favorite_Genre"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Genre"); ax.set_ylabel("Listeners")
    st.pyplot(fig); plt.close(fig)
with right:
    st.subheader("장르별 평균 감상 시간")
    fig, ax = plt.subplots()
    df.groupby("Favorite_Genre")["Listening_Hours"].mean().plot(kind="bar", ax=ax)
    ax.set_xlabel("Genre"); ax.set_ylabel("Average hours")
    st.pyplot(fig); plt.close(fig)

left, right = st.columns(2)
with left:
    st.subheader("선호 감상 시간대")
    fig, ax = plt.subplots()
    df["Preferred_Time"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig); plt.close(fig)
with right:
    st.subheader("주로 사용하는 기기")
    fig, ax = plt.subplots()
    df["Device"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig); plt.close(fig)

st.header("3. Machine Learning")
features = ["Age", "Listening_Hours", "Preferred_Time", "Mood", "Device"]
X, y = df[features], df["Favorite_Genre"]
numeric = ["Age", "Listening_Hours"]
categorical = ["Preferred_Time", "Mood", "Device"]
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}
trained, scores = {}, {}
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    trained[name] = pipe
    scores[name] = accuracy_score(y_test, pipe.predict(X_test))

st.subheader("모델 성능 비교")
st.dataframe(pd.DataFrame({"Model": list(scores), "Accuracy": [round(v, 3) for v in scores.values()]}))
best_name = max(scores, key=scores.get)
st.success(f"테스트 데이터에서 더 높은 정확도를 보인 모델: {best_name} ({scores[best_name]:.1%})")
chosen = st.selectbox("분류 결과를 확인할 모델", list(trained))
preds = trained[chosen].predict(X_test)
report = classification_report(y_test, preds, output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).transpose().round(3))

st.header("4. Predict Your Favorite Genre")
with st.form("prediction_form"):
    age = st.slider("나이", 13, 70, 20)
    hours = st.slider("하루 평균 음악 감상 시간", 0.0, 12.0, 2.0, 0.5)
    preferred_time = st.selectbox("주로 음악을 듣는 시간", sorted(df["Preferred_Time"].unique()))
    mood = st.selectbox("음악을 들을 때의 기분", sorted(df["Mood"].unique()))
    device = st.selectbox("주로 사용하는 기기", sorted(df["Device"].unique()))
    submitted = st.form_submit_button("선호 장르 예측")
if submitted:
    sample = pd.DataFrame([{"Age": age, "Listening_Hours": hours, "Preferred_Time": preferred_time, "Mood": mood, "Device": device}])
    result = trained[best_name].predict(sample)[0]
    st.info(f"예측된 선호 음악 장르: **{result}**")

st.caption("※ CSV는 프로젝트 연습용으로 만든 가상 예시 데이터이며 실제 설문 결과가 아닙니다.")
