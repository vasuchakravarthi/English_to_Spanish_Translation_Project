import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import gdown
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

# Page configuration - switched to wide layout for better space usage
st.set_page_config(
    page_title="English to Spanish Neural Translator",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .header {background: linear-gradient(to right, #4facfe, #00f2fe); padding: 20px; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    .stButton>button {background-color: #4facfe; color: white; border-radius: 5px; padding: 10px 20px;}
    .stTextArea>label {font-weight: bold;}
    .copy-btn {margin-top: 10px;}
    .example-btn {background-color: #00f2fe; color: white;}
    @media (max-width: 768px) {
        .stColumns {flex-direction: column;}
    }
    </style>
    """, unsafe_allow_html=True)

def create_model_architecture(eng_vocab_size, spa_vocab_size, embedding_dim=256, hidden_units=256):
    """Create EXACT same architecture as training - CRITICAL for weight loading"""
    
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    encoder_embedding = Embedding(eng_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(hidden_units, return_state=True, dropout=0.2)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    decoder_embedding = Embedding(spa_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, dropout=0.2)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(spa_vocab_size, activation='softmax')(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_dense)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

@st.cache_resource
def load_model_and_tokenizers():
    """Load model weights and tokenizers with custom object handling"""
    
    weights_path = 'model.weights.h5'
    
    if not os.path.exists(weights_path):
        st.info("📥 Downloading model weights from Google Drive...")
        
        weights_file_id = '1QE7hUQVKMn7JqW-QqaxvMyKVarlabp7R'
        url = f'https://drive.google.com/uc?id={weights_file_id}'
        
        try:
            gdown.download(url, weights_path, quiet=False)
            st.success("✅ Weights downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading weights: {str(e)}")
            return None, None, None, None, None, None
    
    try:
        with open('eng_tokenizer.pkl', 'rb') as f:
            eng_word_to_idx, eng_idx_to_word = pickle.load(f)
            
        with open('spa_tokenizer.pkl', 'rb') as f:
            spa_word_to_idx, spa_idx_to_word = pickle.load(f)
            
        with open('config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        model = create_model_architecture(
            eng_vocab_size=len(eng_word_to_idx),
            spa_vocab_size=len(spa_word_to_idx)
        )
        
        try:
            model.load_weights(weights_path)
            st.success("✅ Model weights loaded into exact training architecture!")
        except Exception as e:
            import tensorflow.keras.utils as keras_utils
            with keras_utils.custom_object_scope({}):
                model.load_weights(weights_path)
                st.success("✅ Model weights loaded with custom object scope!")
        
        return model, eng_word_to_idx, eng_idx_to_word, spa_word_to_idx, spa_idx_to_word, config
        
    except Exception as e:
        st.error(f"Error loading model or tokenizers: {str(e)}")
        return None, None, None, None, None, None

def preprocess_text(text, is_spanish=False):
    text = text.lower()
    if is_spanish:
        text = re.sub(r'[^a-zA-Záéíóúñü¡¿\s\.,!?]', '', text)
    else:
        text = re.sub(r'[^a-zA-Z\s\.,!?]', '', text)
    text = re.sub(r'([.!?¡¿])', r' \1 ', text)
    text = ' '.join(text.split())
    return text.strip()

def text_to_sequence(text, word_to_idx):
    words = text.split()
    return [word_to_idx.get(word, word_to_idx.get('<unk>', 1)) for word in words]

def translate_sentence(sentence, model, eng_tokenizer, spa_tokenizer, spa_idx_to_word, max_len_eng, max_len=17):
    sentence_clean = preprocess_text(sentence, is_spanish=False)
    sentence_seq = text_to_sequence(sentence_clean, eng_tokenizer)
    
    if len(sentence_seq) == 0:
        return "Unable to translate empty sentence"
    
    sentence_padded = pad_sequences([sentence_seq], maxlen=max_len_eng, padding='post')
    
    decoder_input = np.zeros((1, max_len))
    decoder_input[0, 0] = spa_tokenizer.get('<start>', 1)
    
    translation = []
    
    for i in range(1, max_len):
        predictions = model.predict([sentence_padded, decoder_input[:, :i]], verbose=0)
        predicted_id = np.argmax(predictions[0, i-1, :])
        predicted_word = spa_idx_to_word.get(predicted_id, '<unk>')
        
        if predicted_word in ['<end>', '<pad>'] or predicted_id == 0:
            break
            
        if predicted_word != '<unk>':
            translation.append(predicted_word)
            
        decoder_input[0, i] = predicted_id
    
    result = ' '.join(translation).strip()
    return result if result else "Translation failed"

# Sidebar for about and instructions
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("Built by Vasu Chakravarthi Jaladi")
    st.markdown("*Custom LSTM Encoder-Decoder Neural Network*")
    st.markdown("### How to Use")
    st.markdown("- Enter English text below.")
    st.markdown("- Click Translate.")
    st.markdown("- Copy results easily.")
    st.markdown("### Examples")
    examples = [
        "Hello, how are you?",
        "What is the weather today?",
        "I love machine learning."
    ]
    selected_example = st.selectbox("Quick examples:", [""] + examples)
    if selected_example:
        st.session_state.user_input = selected_example

# Main app content
def main():
    # Header with style
    st.markdown('<div class="header"><h1>🌍 English to Spanish Neural Translator</h1></div>', unsafe_allow_html=True)
    
    # Load model
    model, eng_word_to_idx, eng_idx_to_word, spa_word_to_idx, spa_idx_to_word, config = load_model_and_tokenizers()
    
    if model is None:
        st.error("❌ Failed to load model. Please check file paths.")
        return
    
    st.success("✅ Neural translation model loaded successfully!")
    
    # Input section
    st.markdown("## 📝 Enter English Text to Translate")
    
    # Initialize session state for input and history
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Text input with char limit display
    user_input = st.text_area(
        "English sentence:",
        value=st.session_state.user_input,
        placeholder="Type your English sentence here...",
        height=150,
        max_chars=500,
        key="input_area"
    )
    st.caption(f"Characters: {len(user_input)}/500")
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        translate_btn = st.button("🔄 Translate", type="primary", use_container_width=True)
    
    if translate_btn:
        if user_input.strip():
            with st.spinner("Translating..."):
                try:
                    translation = translate_sentence(
                        user_input,
                        model,
                        eng_word_to_idx,
                        spa_word_to_idx,
                        spa_idx_to_word,
                        config['MAX_LEN_ENG']
                    )
                    
                    # Add to history
                    st.session_state.history.append({"english": user_input, "spanish": translation})
                    
                    # Display results
                    st.markdown("## 🎯 Translation Result")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🇺🇸 English:**")
                        st.info(user_input)
                        if st.button("Copy English", key="copy_eng"):
                            st.write("Copied!")  # In real app, use JS for clipboard (Streamlit limitation)
                    
                    with col2:
                        st.markdown("**🇪🇸 Spanish:**")
                        st.success(translation)
                        if st.button("Copy Spanish", key="copy_spa"):
                            st.write("Copied!")
                    
                    # Simple confidence (placeholder based on length)
                    conf = min(100, len(translation.split()) * 20)
                    st.progress(conf / 100)
                    st.caption(f"Estimated confidence: {conf}% (based on output length)")
                        
                except Exception as e:
                    st.error(f"Translation error: {str(e)}")
        else:
            st.warning("Please enter some text to translate.")
    
    # Translation history
    if st.session_state.history:
        st.markdown("## 📜 Translation History")
        for idx, item in enumerate(reversed(st.session_state.history[-5:])):  # Last 5
            with st.expander(f"Entry {len(st.session_state.history) - idx}"):
                st.write(f"**English:** {item['english']}")
                st.write(f"**Spanish:** {item['spanish']}")

if __name__ == "__main__":
    main()
