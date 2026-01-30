import streamlit as st
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
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
    
    /* Hide Streamlit branding - REMOVED per requirement (6) */
    /* #MainMenu {visibility: hidden;} */
    /* footer {visibility: hidden;} */
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load GPT-2 model and tokenizer"""
    model_name = "gpt2"  # Using the smallest GPT-2 model (117M parameters)
    
    # Limit CPU threads to reduce resource usage (requirement 4)
    torch.set_num_threads(2)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set pad token
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    except Exception as e:
        # Make failure explicit (requirement 7) - no demo mode
        error_msg = str(e)[:200]  # Truncate but keep useful details
        st.error(f"❌ Failed to load GPT-2 model. Check requirements.txt and Streamlit build logs.\n\nError: {error_msg}")
        st.stop()
        return None, None  # Never reached due to st.stop()

def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9, repetition_penalty=1.1):
    """Generate response from GPT-2"""
    # Requirement 3b: Truncate prompt to fit GPT-2 context (1024 tokens)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Keep last 900 tokens to leave room for generation
    if input_ids.shape[1] > 900:
        input_ids = input_ids[:, -900:]
    
    # Generate with requirement 3 (max_new_tokens) and requirement 5 (repetition controls)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,  # Requirement 3: use max_new_tokens instead of max_length
            temperature=temperature,
            top_p=top_p,
            top_k=50,  # Requirement 5: stabilize sampling
            repetition_penalty=repetition_penalty,  # Requirement 5: reduce loops
            no_repeat_ngram_size=3,  # Requirement 5: prevent repetitive n-grams
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
        
        # Requirement 3: Renamed "Response Length" to "Max new tokens"
        # Requirement 4: Default set to 100 (reasonable for Community Cloud)
        max_new_tokens = st.slider(
            "Max new tokens",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
            help="Maximum number of new tokens to generate"
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
        
        # Requirement 5: Add anti-repeat slider
        repetition_penalty = st.slider(
            "Anti-repeat penalty",
            min_value=1.0,
            max_value=2.0,
            value=1.1,
            step=0.1,
            help="Higher values reduce repetition (default 1.1)"
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
    
    # Requirement 2: Store model/tokenizer in session_state to avoid reloading spinner
    if "gpt2_model" not in st.session_state or "gpt2_tokenizer" not in st.session_state:
        # First load - show spinner
        with st.spinner("⏳ Loading GPT-2 from 2019... Please wait..."):
            model, tokenizer = load_model()
            st.session_state["gpt2_model"] = model
            st.session_state["gpt2_tokenizer"] = tokenizer
    else:
        # Subsequent reruns - reuse from session_state (no spinner)
        model = st.session_state["gpt2_model"]
        tokenizer = st.session_state["gpt2_tokenizer"]
    
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
        # Requirement 4: Use last 3 turns (6 messages = 3 user + 3 bot) instead of 5
        conversation = ""
        for msg in st.session_state.messages[-6:]:
            if msg["role"] == "user":
                conversation += f"User: {msg['content']}\n"
            else:
                conversation += f"GPT-2: {msg['content']}\n"
        
        conversation += "GPT-2:"
        
        # Generate response
        with st.spinner("🤔 GPT-2 is thinking..."):
            # Requirement 3: No manual prompt_length + max_length logic
            # Just pass max_new_tokens directly
            response = generate_response(
                model, 
                tokenizer, 
                conversation,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
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
