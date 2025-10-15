import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import gdown
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import get_custom_objects

# Add compatibility for older TensorFlow models
def custom_not_equal(x, y):
    """Custom NotEqual layer for backward compatibility"""
    return tf.not_equal(x, y)

# Register the custom object
get_custom_objects().update({'NotEqual': custom_not_equal})

# Page configuration
st.set_page_config(
    page_title="English to Spanish Neural Translator",
    page_icon="🌍",
    layout="centered"
)

@st.cache_resource
def load_model_and_tokenizers():
    """Download and load model from Google Drive with tokenizers"""
    
    model_path = 'simple_translation_model.h5'
    
    # Download model from Google Drive if not exists
    if not os.path.exists(model_path):
        st.info("📥 Downloading neural translation model from Google Drive... (first time only)")
        
        file_id = '1FeUEj87a03AU06b9HiL57xCVOn2m4EPQ'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {str(e)}")
            return None, None, None, None, None, None
    
    try:
        # Load model with custom objects for compatibility
        custom_objects = {
            'NotEqual': custom_not_equal,
            'not_equal': tf.not_equal
        }
        
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False  # Skip compilation to avoid issues
        )
        
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load tokenizers  
        with open('eng_tokenizer.pkl', 'rb') as f:
            eng_word_to_idx, eng_idx_to_word = pickle.load(f)
            
        with open('spa_tokenizer.pkl', 'rb') as f:
            spa_word_to_idx, spa_idx_to_word = pickle.load(f)
            
        # Load configuration
        with open('config.pkl', 'rb') as f:
            config = pickle.load(f)
            
        return model, eng_word_to_idx, eng_idx_to_word, spa_word_to_idx, spa_idx_to_word, config
        
    except Exception as e:
        st.error(f"Error loading model or tokenizers: {str(e)}")
        st.info("💡 If this persists, the model may need to be retrained with TensorFlow 2.20.0")
        return None, None, None, None, None, None

def preprocess_text(text, is_spanish=False):
    """Clean and preprocess text"""
    text = text.lower()
    if is_spanish:
        text = re.sub(r'[^a-zA-Záéíóúñü¡¿\s\.,!?]', '', text)
    else:
        text = re.sub(r'[^a-zA-Z\s\.,!?]', '', text)
    text = re.sub(r'([.!?¡¿])', r' \1 ', text)
    text = ' '.join(text.split())
    return text.strip()

def text_to_sequence(text, word_to_idx):
    """Convert text to numbers"""
    words = text.split()
    return [word_to_idx.get(word, word_to_idx.get('<unk>', 1)) for word in words]

def translate_sentence(sentence, model, eng_tokenizer, spa_tokenizer, spa_idx_to_word, max_len_eng, max_len=17):
    """Translate English sentence to Spanish"""
    
    # Preprocess input
    sentence_clean = preprocess_text(sentence, is_spanish=False)
    sentence_seq = text_to_sequence(sentence_clean, eng_tokenizer)
    
    if len(sentence_seq) == 0:
        return "Unable to translate empty sentence"
    
    sentence_padded = pad_sequences([sentence_seq], maxlen=max_len_eng, padding='post')
    
    # Initialize decoder
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

# Main app
def main():
    st.title("🌍 English to Spanish Neural Translator")
    st.markdown("### Built by Vasu Chakravarthi Jaladi")
    st.markdown("*Custom LSTM Encoder-Decoder Neural Network*")
    
    # Load model and tokenizers
    model, eng_word_to_idx, eng_idx_to_word, spa_word_to_idx, spa_idx_to_word, config = load_model_and_tokenizers()
    
    if model is None:
        st.error("❌ Failed to load model. Please check file paths.")
        return
    
    st.success("✅ Neural translation model loaded successfully!")
    
    # Input section
    st.markdown("## 📝 Enter English Text to Translate")
    
    # Text input
    user_input = st.text_area(
        "English sentence:",
        placeholder="Type your English sentence here...",
        height=100
    )
    
    # Example sentences
    if st.button("Try Example Sentences"):
        examples = [
            "Hello, how are you?",
            "I am very happy.",
            "Where is the bathroom?",
            "Thank you very much.",
            "I want to eat pizza."
        ]
        user_input = st.selectbox("Select an example:", examples)
    
    # Translation button
    if st.button("🔄 Translate", type="primary"):
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
                    
                    # Display results
                    st.markdown("## 🎯 Translation Result")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🇺🇸 English:**")
                        st.info(user_input)
                    
                    with col2:
                        st.markdown("**🇪🇸 Spanish:**")
                        st.success(translation)
                        
                except Exception as e:
                    st.error(f"Translation error: {str(e)}")
        else:
            st.warning("Please enter some text to translate.")
    
    # Model information
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    
    if config:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("English Vocabulary", f"{config['ENG_VOCAB_SIZE']:,}")
        
        with col2:
            st.metric("Spanish Vocabulary", f"{config['SPA_VOCAB_SIZE']:,}")
            
        with col3:
            st.metric("Architecture", "LSTM Encoder-Decoder")

if __name__ == "__main__":
    main()
