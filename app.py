import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ================= MODEL =================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):

        super(MultiHeadSelfAttention, self).__init__()
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads

        self.query_dense = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_dense = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_dense = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def attention(self, query, key, value):
        score = torch.matmul(query, key.transpose(-2, -1))
        dim_key = torch.tensor(key.shape[-1], dtype=torch.float32)
        scaled_score = score / torch.sqrt(dim_key + 1e-9)
        weights = torch.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, _ = self.attention(query, key, value)
        attention = attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = attention.view(batch_size, -1, self.embed_dim)
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.2):
        super(TransformerBlock, self).__init__()
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


def proposed_model(X_train_shape, num_classes):
    embed_dim = 1024  # Embedding size for each token
    num_heads = 128  # Number of attention heads
    ff_dim = 256  # Hidden layer size in feed forward network inside transformer

    class Model(nn.Module):
      def __init__(self):

        super(Model, self).__init__()
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)
        self.reshape = nn.Linear(X_train_shape[-1], 1024*2)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.batch_norm = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.dense1 = nn.Linear(embed_dim, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(128, 128)
        self.dropout3 = nn.Dropout(0.2)
        self.output_layer = nn.Linear(128, num_classes)

      def forward(self, x):
          x = x.view(-1, 2, 1024)
          x = self.transformer_block(x)
          x = self.global_avg_pool(x.transpose(1, 2)).squeeze(-1)
          x = self.batch_norm(x)
          x = self.dropout1(torch.selu(self.dense1(x)))
          x = self.dropout2(torch.selu(self.dense2(x)))
          x = self.dropout3(x)
          x = torch.softmax(self.output_layer(x), dim=-1)
          return x
    return Model()




# ====== CLASS LABELS ======
classes = ["32PSK", "16APSK", "32QAM", "FM", "GMSK", "32APSK",
           "OQPSK", "8ASK", "BPSK", "8PSK", "AM-SSB-SC", "4ASK",
           "16PSK", "64APSK", "128QAM", "128APSK", "AM-DSB-SC",
           "AM-SSB-WC", "64QAM", "QPSK", "256QAM", "AM-DSB-WC",
           "OOK", "16QAM"]

# ====== LOAD MODEL FROM CHECKPOINT ======
@st.cache_resource
def load_model():
    model = proposed_model((1024, 2), 24)

    checkpoint = torch.load("checkpoint_epoch_130.pth", map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    return model

model = load_model()

# ====== STREAMLIT UI ======
# ================= UI =================

# 🎨 STYLE
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    font-family: 'Segoe UI', sans-serif;
}

/* FORCE TEXT WHITE EVERYWHERE */
* {
    color: white !important;
}

/* RADIO LABEL TEXT (THIS IS THE KEY FIX) */
div[role="radiogroup"] label span {
    color: white !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

/* Radio circle visibility */
div[role="radiogroup"] input {
    accent-color: #00ffe7 !important;
}

/* File uploader text */
section[data-testid="stFileUploader"] span {
    color: white !important;
}

/* Button styling */
.stButton > button {
    background-color: #1f2933 !important;   /* dark blue-grey */
    color: white !important;
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: 600;
    border: 1px solid #00ffe7;
}

/* HOVER EFFECT */
.stButton > button:hover {
    background-color: #111827 !important;
    color: #00ffe7 !important;
    border: 1px solid #00ffe7;
}

/* CLICK EFFECT */
.stButton > button:active {
    background-color: #020617 !important;
}


/* Headings */
h1, h2, h3 {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# 🧠 HEADER
st.markdown("# 📡 RF Signal Classification")
st.markdown("### Transformer-based Modulation Recognition")

# 📦 INPUT CARD
st.markdown('<div class="card">', unsafe_allow_html=True)

option = st.radio("Select Input Type:", ["Random Signal", "Upload .npy File"])

if option == "Random Signal":
    signal = np.random.randn(1024, 2)
else:
    uploaded_file = st.file_uploader("Upload .npy file (shape: 1024 x 2)")
    if uploaded_file is not None:
        signal = np.load(uploaded_file)
    else:
        st.stop()

st.markdown('</div>', unsafe_allow_html=True)

# 📊 SIGNAL VISUAL
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("IQ Signal Visualization")

fig, ax = plt.subplots()
ax.plot(signal[:, 0], label="In-phase (I)", linewidth=1.5)
ax.plot(signal[:, 1], label="Quadrature (Q)", linewidth=1.5)
ax.legend()
ax.set_facecolor("#111111")
fig.patch.set_facecolor("#111111")

st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# 🔮 PREDICTION
st.markdown('<div class="card">', unsafe_allow_html=True)

if st.button("Predict Modulation"):

    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(x)

        pred = torch.argmax(output, dim=1).item()
        confidence = output[0][pred].item()

    # BIG RESULT
    st.markdown(f"## 🟢 {classes[pred]}")
    st.write(f"### Confidence: {confidence*100:.2f}%")

    # TOP 3
    st.subheader("Top Predictions")

    top3 = torch.topk(output, 3)

    for i in range(3):
        idx = top3.indices[0][i].item()
        conf = top3.values[0][i].item()
        st.progress(conf)
        st.write(f"{classes[idx]} — {conf*100:.2f}%")

st.markdown('</div>', unsafe_allow_html=True)