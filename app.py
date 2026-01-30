import streamlit as st
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
import random
import time
import html

# Page config
st.set_page_config(
    page_title="RetroGPT-2 Chat",
    page_icon="💾",
    layout="wide"
)

# Retro 90s/2000s CSS styling
st.markdown("""
<style>
    /* Main background - classic Windows 95/98 teal */
    .stApp {
        background: linear-gradient(180deg, #008080 0%, #006666 100%);
        font-family: 'Courier New', monospace;
    }
    
    /* Title styling - retro web 1.0 */
    h1 {
        color: #00FF00;
        text-align: center;
        font-family: monospace;
        font-size: 4em !important;
        text-shadow: 3px 3px 0px #000000, 6px 6px 0px #333333;
        animation: blink 2s infinite;
        letter-spacing: 3px;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Chat container - old-school window */
    .chat-container {
        background: #C0C0C0;
        border: 4px outset #FFFFFF;
        border-radius: 0;
        padding: 15px;
        margin: 20px 0;
        box-shadow: 8px 8px 0px rgba(0,0,0,0.5);
        font-family: 'Courier New', monospace;
    }
    
    /* Message styling */
    .user-message {
        background: #FFFF99;
        border: 2px solid #000000;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        color: #000000;
        box-shadow: 3px 3px 0px rgba(0,0,0,0.3);
    }
    
    .bot-message {
        background: #99CCFF;
        border: 2px solid #000000;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        color: #000000;
        box-shadow: 3px 3px 0px rgba(0,0,0,0.3);
    }
    
    /* Input box styling */
    .stTextInput > div > div > input {
        background: #FFFFFF;
        border: 2px inset #808080;
        font-family: 'Courier New', monospace;
        color: #000000;
        border-radius: 0;
    }
    
    /* Button styling - classic Windows button */
    .stButton > button {
        background: #C0C0C0;
        border: 2px outset #FFFFFF;
        border-radius: 0;
        color: #000000;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        padding: 8px 24px;
        box-shadow: 3px 3px 0px rgba(0,0,0,0.3);
    }
    
    .stButton > button:hover {
        background: #D0D0D0;
        border: 2px inset #FFFFFF;
    }
    
    .stButton > button:active {
        border: 2px inset #808080;
        box-shadow: 1px 1px 0px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #008080;
        border-right: 4px solid #000000;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #FFFF00;
        font-family: monospace;
        text-shadow: 2px 2px 0px #000000;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #FFFFFF;
        font-family: 'Courier New', monospace;
    }
    
    /* Info box */
    .info-box {
        background: #FFFF99;
        border: 3px double #000000;
        padding: 15px;
        margin: 20px 0;
        font-family: 'Courier New', monospace;
        color: #000000;
    }
    
    /* Retro counter/badge */
    .retro-badge {
        background: #FF0000;
        color: #FFFFFF;
        padding: 5px 10px;
        border: 2px solid #000000;
        display: inline-block;
        font-family: monospace;
        font-size: 1.5em;
        box-shadow: 3px 3px 0px rgba(0,0,0,0.5);
    }
    
    /* Horizontal rule - retro style */
    hr {
        border: 2px dotted #000000;
        margin: 20px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load GPT-2 model and tokenizer"""
    model_name = "gpt2"  # Using the smallest GPT-2 model (117M parameters)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set pad token
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    except Exception as e:
        st.warning(f"⚠️ Could not load GPT-2 model: {str(e)[:100]}... Using demo mode.")
        return None, None

def generate_response(model, tokenizer, prompt, max_length=100, temperature=0.8, top_p=0.9):
    """Generate response from GPT-2"""
    # Demo/fallback mode if model not available
    if model is None or tokenizer is None:
        time.sleep(0.5)  # Simulate processing
        demo_responses = [
            "That's an interesting point. In 2019, language models like me were still in their early stages.",
            "I appreciate your message! Back in my time, I could only predict the next word, not have real conversations.",
            "Thanks for chatting! Remember, I'm from 2019 - before RLHF and instruction tuning existed.",
            "Fascinating! As a 2019 model, my responses might seem simple compared to today's AI.",
            "I hear you! In February 2019, when I was released, this level of interaction was cutting-edge.",
            "Interesting question. My 117M parameters were considered large in 2019, but tiny by today's standards!",
            "I see what you mean. Back then, we didn't have the sophisticated training methods that modern models use.",
            "That's a good observation. I'm just doing next-token prediction, nothing fancy like ChatGPT!",
        ]
        return random.choice(demo_responses)
    
    # Real GPT-2 generation
    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # Decode
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from response
    if len(response) > len(prompt):
        response = response[len(prompt):].strip()
    else:
        response = response.strip()
    
    return response

def main():
    # Title with retro styling
    st.markdown("<h1>💾 RetroGPT-2 Chat 💾</h1>", unsafe_allow_html=True)
    
    # Retro info banner
    st.markdown("""
    <div class="info-box">
        <center>
        <b>⚡ Welcome to the Time Machine! ⚡</b><br>
        Step back to 2019 and chat with GPT-2 (117M parameters)<br>
        <i>Experience where it all began... before ChatGPT changed everything!</i>
        </center>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")
        
        max_length = st.slider(
            "Response Length",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
            help="Maximum length of generated response"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        top_p = st.slider(
            "Top P",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Nucleus sampling parameter"
        )
        
        st.markdown("---")
        st.markdown("### 📜 About GPT-2")
        st.markdown("""
        Released in **February 2019** by OpenAI
        
        **Specs:**
        - 117M parameters (gpt2)
        - Trained on WebText
        - No RLHF or instruction tuning
        - Pure next-token prediction
        
        **Fun Fact:**
        This was considered so powerful that OpenAI initially hesitated to release it! 🤖
        """)
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load model
    with st.spinner("⏳ Loading GPT-2 from 2019... Please wait..."):
        model, tokenizer = load_model()
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        # Escape HTML to prevent XSS
        escaped_content = html.escape(message["content"])
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><b>👤 You:</b> {escaped_content}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><b>🤖 GPT-2:</b> {escaped_content}</div>', 
                       unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input(
        "Type your message:",
        key="user_input",
        placeholder="Enter your message here..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        send_button = st.button("📤 Send")
    
    # Handle user input
    if send_button and user_input:
        # Validate input
        if not user_input.strip():
            st.warning("⚠️ Please enter a message.")
            st.stop()
        
        if len(user_input) > 500:
            st.warning("⚠️ Message too long. Please keep it under 500 characters.")
            st.stop()
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Build conversation context
        conversation = ""
        for msg in st.session_state.messages[-5:]:  # Use last 5 messages for context
            if msg["role"] == "user":
                conversation += f"User: {msg['content']}\n"
            else:
                conversation += f"GPT-2: {msg['content']}\n"
        
        conversation += "GPT-2:"
        
        # Generate response
        with st.spinner("🤔 GPT-2 is thinking..."):
            # Calculate max_length properly
            if tokenizer is not None:
                prompt_length = len(tokenizer.encode(conversation))
                total_max_length = prompt_length + max_length
            else:
                total_max_length = max_length
            
            response = generate_response(
                model, 
                tokenizer, 
                conversation,
                max_length=total_max_length,
                temperature=temperature,
                top_p=top_p
            )
        
        # Add bot response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to display new messages
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <center>
    <div class="retro-badge">Est. 2019</div>
    <p style="color: #FFFFFF; font-family: monospace; font-size: 1.2em;">
    🌐 Powered by GPT-2 • Built with Streamlit 🌐<br>
    <i>A nostalgic journey through the early days of large language models</i>
    </p>
    </center>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
