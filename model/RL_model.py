import numpy as np
import torch
import torch.nn as nn

DIAGNOSIS_TREATMENT_MAP = {
    "Asthma": [
        "Inhaler therapy",
        "Bronchodilator",
        "Steroid inhaler"
    ],
    "Type 2 Diabetes": [
        "Lifestyle modification",
        "Oral medication",
        "Insulin therapy"
    ],
    "Hypertension": [
        "Lifestyle modification",
        "ACE inhibitors",
        "Beta blockers"
    ]
}

# =========================================================
# 1. Model Definitions (Reusable)
# =========================================================

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

# =========================================================
# 2. Driver Function (Pure Inference Logic)
# =========================================================

def recommend_treatment(
    age,
    gender,
    diagnosis,
    heart_rate,
    respiratory_rate,
    oxygen_saturation,
    encoder,
    model,
    encoders
):
    """
    Stateless inference function.
    All model objects are passed explicitly.
    """

    # Encode categorical inputs
    gender_enc = encoders["Gender"].transform([gender])[0]
    diagnosis_enc = encoders["Diagnosis"].transform([diagnosis])[0]

    # Build state vector
    state = np.array([
        age,
        gender_enc,
        diagnosis_enc,
        heart_rate,
        respiratory_rate,
        oxygen_saturation
    ], dtype=np.float32)

    # Model inference
    with torch.no_grad():
        z = encoder(torch.FloatTensor(state))
        q_vals = model(z)

    # ---- Diagnosis-aware action masking ----
    allowed_treatments = DIAGNOSIS_TREATMENT_MAP.get(diagnosis, [])

    allowed_indices = [
        encoders["TreatmentPlan"].transform([t])[0]
        for t in allowed_treatments
        if t in encoders["TreatmentPlan"].classes_
    ]

    masked_q = q_vals.clone()
    for i in range(len(masked_q)):
        if i not in allowed_indices:
            masked_q[i] = -1e9  # disable invalid treatments

    action = torch.argmax(masked_q).item()


    # Decode treatment
    treatment = encoders["TreatmentPlan"].inverse_transform([action])[0]

    return {
        "RecommendedTreatment": treatment,
        "QValues": q_vals.numpy().tolist()
    }
