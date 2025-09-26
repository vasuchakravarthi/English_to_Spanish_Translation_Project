import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re

# Configure page
st.set_page_config(
    page_title="English-Spanish Neural Translator",
    page_icon="🌍",
    layout="centered"
)

# Title and description
st.title("🌍 Neural Machine Translation")
st.subheader("English → Spanish Translation")
st.markdown("**Achieved 35.36 BLEU Score** - Commercial-grade neural translator")

# Load model and tokenizers (cached for performance)
@st.cache_resource
def load_translation_model():
    """Load the trained model and tokenizers"""
    try:
        # Load your trained model
        model = tf.keras.models.load_model('simple_translation_model.h5', compile=False)
        
        # Load tokenizers (you'll need to save these from your training)
        with open('eng_tokenizer.pkl', 'rb') as f:
            eng_tokenizer = pickle.load(f)
        with open('spa_tokenizer.pkl', 'rb') as f:
            spa_tokenizer = pickle.load(f)
            
        return model, eng_tokenizer, spa_tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def preprocess_text(text):
    """Clean and preprocess input text"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\.\,\?\!\-\']', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def translate_text(english_text, model, eng_tokenizer, spa_tokenizer):
    """Translate English text to Spanish"""
    if not english_text.strip():
        return "Please enter some text to translate."
    
    try:
        # Preprocess input
        clean_text = preprocess_text(english_text)
        
        # Tokenize input
        input_seq = eng_tokenizer.texts_to_sequences([clean_text])
        
        if not input_seq[0]:  # Empty sequence
            return "Could not process the input text. Please try different words."
        
        # Pad sequence
        max_len = 20  # Use same max length as training
        input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=max_len, padding='post')
        
        # Get prediction
        prediction = model.predict(input_seq, verbose=0)
        
        # Convert prediction to text
        predicted_ids = np.argmax(prediction[0], axis=1)
        
        # Remove padding and end tokens
        predicted_words = []
        for word_id in predicted_ids:
            if word_id == 0:  # Skip padding
                continue
            if word_id in spa_tokenizer.index_word:
                word = spa_tokenizer.index_word[word_id]
                if word == '<end>':
                    break
                if word != '<start>':
                    predicted_words.append(word)
        
        return ' '.join(predicted_words) if predicted_words else "Translation not available for this input."
        
    except Exception as e:
        return f"Translation error: {str(e)}"

# Load model
model, eng_tokenizer, spa_tokenizer = load_translation_model()

if model is not None:
    # Main interface
    st.markdown("---")
    
    # Input section
    st.markdown("### 📝 Enter English Text")
    english_input = st.text_area(
        "Type your English sentence here:",
        placeholder="e.g., Hello, how are you?",
        height=100,
        key="english_input"
    )
    
    # Translation button
    if st.button("🔄 Translate to Spanish", type="primary"):
        if english_input:
            with st.spinner("Translating..."):
                spanish_output = translate_text(english_input, model, eng_tokenizer, spa_tokenizer)
            
            # Results section
            st.markdown("### 🎯 Translation Result")
            st.success(spanish_output)
            
            # Copy button functionality
            st.markdown(f"**Spanish:** {spanish_output}")
            
        else:
            st.warning("Please enter some English text to translate.")
    
    # Example translations
    st.markdown("---")
    st.markdown("### 💡 Try These Examples:")
    
    examples = [
        "Hello, how are you?",
        "I love machine learning.",
        "The weather is beautiful today.",
        "Thank you for your help."
    ]
    
    col1, col2 = st.columns(2)
    for i, example in enumerate(examples):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(f"'{example}'", key=f"example_{i}"):
                st.session_state.english_input = example
    
    # Model info
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("BLEU Score", "35.36", "Commercial Grade")
    with col2:
        st.metric("Architecture", "Seq2Seq", "with Attention")
    with col3:
        st.metric("Framework", "TensorFlow", "2.20.0")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <p>🚀 Built by <strong>Vasu Chakravarthi Jaladi</strong> | 
        <a href='https://github.com/vasuchakravarthi/English_to_Spanish_Translation_Project' target='_blank'>GitHub</a> | 
        Neural Machine Translation System</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

else:
    st.error("❌ Could not load the translation model. Please check if all model files are present.")
    st.info("Required files: simple_translation_model.h5, eng_tokenizer.pkl, spa_tokenizer.pkl")
