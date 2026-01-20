import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Load Dataset

df = pd.read_csv("synthetic_multimodal_patient_dataset.csv")

# Preserve outcome text for clinical interpretation
df["OutcomeText"] = df["Outcome"]


# 2A. Encode Categorical Columns

label_cols = ["Gender", "Diagnosis", "TreatmentPlan", "ActivityLevel", "Outcome"]
encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


# 2B. Inspect Treatment Labels (DEBUG / ANALYSIS)

print("\n================ TREATMENT LABEL INSPECTION ================")

# Number of unique treatments
print("Number of unique TreatmentPlan labels:")
print(df["TreatmentPlan"].nunique())

# Encoded -> original mapping
print("\nEncoded TreatmentPlan labels:")
treatment_encoder = encoders["TreatmentPlan"]

for i, label in enumerate(treatment_encoder.classes_):
    print(f"Encoded {i}  ->  {label}")

# Outcome distribution per treatment
print("\nOutcome distribution per TreatmentPlan:\n")

for a in df["TreatmentPlan"].unique():
    label = treatment_encoder.inverse_transform([a])[0]
    subset = df[df["TreatmentPlan"] == a]

    print(f"Treatment: {label}")
    print(subset["OutcomeText"].value_counts(normalize=True))
    print("-" * 50)

print("===========================================================\n")


# 3. Build State, Action

state_cols = [
    "Age",
    "Gender",
    "Diagnosis",
    "HeartRate",
    "RespiratoryRate",
    "OxygenSaturation"
]

states = df[state_cols].values
actions = df["TreatmentPlan"].values

state_dim = states.shape[1]
num_actions = df["TreatmentPlan"].nunique()


# 4A. Transformer-based Tabular Encoder

class TabularTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=32, num_heads=4):
        super().__init__()

        self.feature_embed = nn.Linear(1, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

        self.output_dim = embed_dim

    def forward(self, x):
        # x: [state_dim]
        x = x.unsqueeze(0).unsqueeze(-1)   # [1, state_dim, 1]
        x = self.feature_embed(x)          # [1, state_dim, embed_dim]
        x = self.transformer(x)            # [1, state_dim, embed_dim]
        x = x.mean(dim=1)                  # [1, embed_dim]
        return x.squeeze(0)                # [embed_dim]


# 4B. Q-Network (Policy Model)

class QNetwork(nn.Module):
    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# 5. Clinical Domain Knowledge

TOLERANCE = {
    "HeartRate": 10,
    "RespiratoryRate": 2,
    "OxygenSaturation": 2,
    "SystolicBP": 10,
    "DiastolicBP": 5
}

def parse_bp(bp):
    s, d = bp.split("/")
    return int(s), int(d)

def deviation_with_tolerance(value, low, high, tol):
    if value < low - tol:
        return (low - tol) - value
    elif value > high + tol:
        return value - (high + tol)
    else:
        return 0


# 6. Clinically Grounded Reward Function

def clinical_reward(row):
    r = 0.0

    r -= deviation_with_tolerance(row["HeartRate"], 60, 100, TOLERANCE["HeartRate"]) * 0.01
    r -= deviation_with_tolerance(row["RespiratoryRate"], 12, 20, TOLERANCE["RespiratoryRate"]) * 0.02
    r -= deviation_with_tolerance(row["OxygenSaturation"], 95, 100, TOLERANCE["OxygenSaturation"]) * 0.05

    sys, dia = parse_bp(row["BloodPressure"])
    r -= deviation_with_tolerance(sys, 90, 120, TOLERANCE["SystolicBP"]) * 0.01
    r -= deviation_with_tolerance(dia, 60, 80, TOLERANCE["DiastolicBP"]) * 0.01

    outcome = row["OutcomeText"]
    if "Improved" in outcome or "Controlled" in outcome:
        r += 1.0
    elif "Stable" in outcome:
        r += 0.5
    else:
        r -= 1.0

    return r


# 7. Counterfactual Safety-Aware Reward (NOVELTY)

def counterfactual_reward(q_values, action, clinical_r, lambda_cf=0.5):
    best_alt = torch.max(q_values).item()
    chosen_val = q_values[action].item()
    penalty = max(0.0, best_alt - chosen_val)
    return clinical_r - lambda_cf * penalty


# 8. Train Function

def train(use_counterfactual=False):
    encoder = TabularTransformer(state_dim)
    model = QNetwork(encoder.output_dim, num_actions)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(model.parameters()),
        lr=1e-3
    )

    loss_fn = nn.MSELoss()

    X_tr, X_te, A_tr, A_te, idx_tr, idx_te = train_test_split(
        states, actions, df.index.values, test_size=0.2, random_state=42
    )

    for epoch in range(40):
        total_loss = 0.0

        for s, a, idx in zip(X_tr, A_tr, idx_tr):
            s = torch.FloatTensor(s)

            z = encoder(s)
            q_vals = model(z)

            row = df.loc[idx]
            c_r = clinical_reward(row)

            reward = (
                counterfactual_reward(q_vals, a, c_r)
                if use_counterfactual else c_r
            )

            target = q_vals.clone().detach()
            target[a] = reward

            loss = loss_fn(q_vals, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss:.3f}")

    return encoder, model, X_te, A_te, idx_te


# 9. Evaluation

def evaluate(encoder, model, X_test, A_test, idx_test):
    rewards = []
    positive = 0

    with torch.no_grad():
        for s, idx in zip(X_test, idx_test):
            s = torch.FloatTensor(s)
            z = encoder(s)
            q_vals = model(z)

            row = df.loc[idx]
            r = clinical_reward(row)
            rewards.append(r)

            if r > 0:
                positive += 1

    return positive / len(X_test), np.mean(rewards)


# 10. Run Experiments

print("\nTraining BASELINE clinical RL...")
base_encoder, base_model, X_te, A_te, idx_te = train(use_counterfactual=False)
base_acc, base_r = evaluate(base_encoder, base_model, X_te, A_te, idx_te)

print("\nTraining COUNTERFACTUAL safety-aware RL...")
cf_encoder, cf_model, _, _, _ = train(use_counterfactual=True)
cf_acc, cf_r = evaluate(cf_encoder, cf_model, X_te, A_te, idx_te)

print("\n=========== RESULTS ===========")
print(f"Baseline Policy Positivity Rate : {base_acc:.3f}")
print(f"Counterfactual Positivity Rate  : {cf_acc:.3f}")
print("--------------------------------")
print(f"Baseline Avg Clinical Reward    : {base_r:.3f}")
print(f"Counterfactual Avg Reward      : {cf_r:.3f}")
print("================================")
