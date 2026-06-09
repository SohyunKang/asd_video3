import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# =========================
# Load
# =========================

df = pd.read_excel("eye_contact_features.xlsx")

# ASD vs 정상군만 사용
df = df[df["group"].isin(["자폐군", "정상군"])].copy()

# group 없는 것 제외
df = df.dropna(subset=["group"]).copy()


# =========================
# No eye contact 처리
# =========================

df["has_eye_contact"] = df["has_eye_contact"].astype(int)

# duration은 eye contact 없으면 0
df["duration_sec"] = df["duration_sec"].fillna(0)

# emotion 없으면 None 범주
df["representative_emotion"] = (
    df["representative_emotion"]
    .fillna("None")
    .replace("No_EyeContact", "None")
)

# start_sec 처리
# eye contact 있는 경우: 실제 start_sec
# eye contact 없는 경우: 늦은 임의의 시간
# video_duration_sec가 없으면 전체 start_sec 최대값 + 1 사용
if "video_duration_sec" in df.columns:
    df["start_sec"] = df["start_sec"].where(
        df["has_eye_contact"] == 1,
        df["video_duration_sec"] + 1
    )
else:
    late_time = df["start_sec"].dropna().max() + 1
    df["start_sec"] = df["start_sec"].where(
        df["has_eye_contact"] == 1,
        late_time
    )

# 혹시 남은 결측 보정
late_time_global = df["start_sec"].dropna().max() + 1
df["start_sec"] = df["start_sec"].fillna(late_time_global)


# =========================
# Feature / label
# =========================

X = df[
    [
        "start_sec",
        "duration_sec",
        "has_eye_contact",
        "representative_emotion",
    ]
].copy()

y = (df["group"] == "자폐군").astype(int)


# =========================
# Preprocessing
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "emotion",
            OneHotEncoder(handle_unknown="ignore"),
            ["representative_emotion"],
        )
    ],
    remainder="passthrough"
)


# =========================
# Model
# =========================

model = Pipeline([
    ("prep", preprocessor),
    ("xgb", XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    ))
])


# =========================
# Train / test
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# Evaluation
# =========================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["정상군", "자폐군"]
    )
)


# =========================
# 5-fold CV
# =========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

auc_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print(
    "\n5-fold ROC-AUC:",
    round(auc_scores.mean(), 4),
    "+/-",
    round(auc_scores.std(), 4)
)