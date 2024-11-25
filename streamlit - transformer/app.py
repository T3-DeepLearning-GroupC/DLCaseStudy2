import streamlit as st
import tensorflow as tf
import numpy as np

# Define the custom PositionalEncoding layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maxlen=200, embed_dim=64, vocab_size=None, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)  # Shape: (maxlen,)
        positions = tf.expand_dims(positions, axis=1)  # Shape: (maxlen, 1)
        encoding = tf.cast(positions, dtype=tf.float32) / tf.cast(self.embed_dim, tf.float32)  # Shape: (maxlen, 1)
        encoding = tf.tile(encoding, [1, self.embed_dim])  # Shape: (maxlen, embed_dim)
        encoding = tf.expand_dims(encoding, axis=0)  # Shape: (1, maxlen, embed_dim)
        return inputs + encoding  # Broadcast addition to match input shape

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim,
            "vocab_size": self.vocab_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Define the custom TransformerBlock layer
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.att.key_dim,
            "num_heads": self.att.num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "rate": self.dropout1.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Function to load the model with custom objects
@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model
    model = load_model(
        'transformer_model.h5',
        custom_objects={
            'PositionalEncoding': PositionalEncoding,
            'TransformerBlock': TransformerBlock
        }
    )
    return model

# Load the model
model = load_model()

# Define the prediction function
def make_prediction(input_data):
    # Example preprocessing logic (adjust as necessary for your model)
    processed_data = np.array([input_data])  # Replace with actual preprocessing steps
    prediction = model.predict(processed_data)
    return prediction

# Streamlit app interface
st.title("Transformer Model Prediction App")

# Input form
input_text = st.text_area("Enter your input text:", "")

if st.button("Predict"):
    if input_text:
        # Example preprocessing: Replace with your preprocessing logic
        input_data = [input_text]  # Adjust to match model input
        try:
            prediction = make_prediction(input_data)
            st.write("Prediction:", prediction)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter some text for prediction.")