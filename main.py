            genre = rng.choice(["발라드", "클래식", "팝"])
        else:
            genre = rng.choice(genres)
        rows.append([time, mood, device, hours, genre])
    return pd.DataFrame(rows, columns=["감상 시간대", "기분", "사용 기기", "감상 시간", "선호 장르"])

df = make_data()
st.caption("※ 아래 데이터는 실제 조사 결과가 아닌 프로그램 작동을 위한 가상 예시 데이터입니다.")

c1, c2, c3 = st.columns(3)
c1.metric("분석 데이터 수", f"{len(df)}명")
c2.metric("평균 감상 시간", f"{df['감상 시간'].mean():.1f}시간")
c3.metric("가장 많은 선호 장르", df["선호 장르"].mode()[0])

st.subheader("1. 음악 감상 데이터")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("2. 감상 습관 시각화")
left, right = st.columns(2)
with left:
    st.write("**선호 장르별 인원**")
    counts = df["선호 장르"].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("선호 장르")
    ax.set_ylabel("인원")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.write("**시간대별 평균 감상 시간**")
    avg = df.groupby("감상 시간대")["감상 시간"].mean().reindex(["아침", "오후", "저녁", "밤"])
    fig, ax = plt.subplots()
    avg.plot(kind="bar", ax=ax)
    ax.set_xlabel("감상 시간대")
    ax.set_ylabel("평균 시간")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("3. 머신러닝으로 선호 장르 예측")
X = pd.get_dummies(df[["감상 시간대", "기분", "사용 기기"]])
X["감상 시간"] = df["감상 시간"]
y = df["선호 장르"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}
scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    scores[name] = accuracy_score(y_test, model.predict(X_test))
st.write("**모델 정확도 비교**")
st.dataframe(pd.DataFrame({"모델": list(scores.keys()), "정확도": [f"{v:.1%}" for v in scores.values()]}), hide_index=True)

st.subheader("4. 나의 음악 취향 예측해 보기")
a, b, c = st.columns(3)
with a:
    selected_time = st.selectbox("주로 듣는 시간대", ["아침", "오후", "저녁", "밤"])
with b:
    selected_mood = st.selectbox("음악을 들을 때 기분", ["행복", "차분함", "우울함", "신남"])
with c:
    selected_device = st.selectbox("주로 사용하는 기기", ["스마트폰", "컴퓨터", "태블릿"])
hours = st.slider("하루 평균 음악 감상 시간", 0.5, 8.0, 2.0, 0.5)
if st.button("선호 장르 예측하기"):
    sample = pd.DataFrame([[selected_time, selected_mood, selected_device, hours]],
                          columns=["감상 시간대", "기분", "사용 기기", "감상 시간"])
    sample_X = pd.get_dummies(sample[["감상 시간대", "기분", "사용 기기"]])
    sample_X["감상 시간"] = sample["감상 시간"]
    sample_X = sample_X.reindex(columns=X.columns, fill_value=0)
    pred = models["Random Forest"].predict(sample_X)[0]
    st.success(f"예측된 선호 장르는 **{pred}** 입니다.")

st.info("이 프로그램은 CSV 파일을 읽지 않으므로 CSV 열 이름 오류가 발생하지 않습니다.")
