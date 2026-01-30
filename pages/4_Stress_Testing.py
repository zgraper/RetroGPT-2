import streamlit as st
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
import time
import html

# Page config
st.set_page_config(
    page_title="RetroGPT-2 Stress Test",
    page_icon="⚡",
    layout="wide"
)

# Retro 90s/2000s CSS styling (matching main app)
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
    
    /* Container styling */
    .test-container {
        background: #C0C0C0;
        border: 4px outset #FFFFFF;
        border-radius: 0;
        padding: 15px;
        margin: 20px 0;
        box-shadow: 8px 8px 0px rgba(0,0,0,0.5);
        font-family: 'Courier New', monospace;
    }
    
    /* Result boxes */
    .result-box {
        background: #FFFF99;
        border: 2px solid #000000;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        color: #000000;
        box-shadow: 3px 3px 0px rgba(0,0,0,0.3);
    }
    
    /* Success box */
    .success-box {
        background: #99FF99;
        border: 2px solid #000000;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        color: #000000;
        box-shadow: 3px 3px 0px rgba(0,0,0,0.3);
    }
    
    /* Error box */
    .error-box {
        background: #FF9999;
        border: 2px solid #000000;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        color: #000000;
        box-shadow: 3px 3px 0px rgba(0,0,0,0.3);
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
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load GPT-2 model and tokenizer"""
    model_name = "gpt2"
    
    # Limit CPU threads to reduce resource usage
    torch.set_num_threads(2)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set pad token
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    except Exception as e:
        error_msg = str(e)[:200]
        st.error(f"❌ Failed to load GPT-2 model.\n\nError: {error_msg}")
        st.stop()

def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8):
    """Generate response from GPT-2"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Keep last 900 tokens to leave room for generation
    if input_ids.shape[1] > 900:
        input_ids = input_ids[:, -900:]
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from response
    prompt_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    if len(response) > len(prompt_decoded):
        response = response[len(prompt_decoded):].strip()
    else:
        response = response.strip()
    
    return response

def run_stress_test(model, tokenizer, test_type, num_iterations):
    """Run various stress tests on the model"""
    results = []
    
    test_prompts = {
        "short": ["Hello", "AI", "The"],
        "medium": [
            "Tell me about artificial intelligence",
            "What is the meaning of life?",
            "Explain quantum computing"
        ],
        "long": [
            "In the distant future, humanity has colonized Mars and established a thriving civilization. Write a story about",
            "The history of computing spans many decades, from the first mechanical calculators to modern quantum computers. Let me tell you about",
            "Climate change is one of the most pressing issues facing our planet today. Scientists around the world are working to"
        ],
        "repetitive": [
            "The cat sat on the mat. The cat sat on the mat. What happened next?",
            "One two three one two three one two three. Continue:",
            "Hello world hello world hello world. What comes next?"
        ]
    }
    
    prompts = test_prompts.get(test_type, test_prompts["medium"])
    
    for i in range(num_iterations):
        prompt = prompts[i % len(prompts)]
        
        try:
            start_time = time.time()
            response = generate_response(model, tokenizer, prompt, max_new_tokens=50)
            elapsed_time = time.time() - start_time
            
            results.append({
                "iteration": i + 1,
                "prompt": prompt,
                "response": response,
                "time": elapsed_time,
                "status": "✅ Success"
            })
        except Exception as e:
            results.append({
                "iteration": i + 1,
                "prompt": prompt,
                "response": f"Error: {str(e)[:100]}",
                "time": 0,
                "status": "❌ Failed"
            })
    
    return results

def main():
    # Title
    st.markdown("<h1>⚡ Stress Test GPT-2 ⚡</h1>", unsafe_allow_html=True)
    
    # Info banner
    st.markdown("""
    <div class="info-box">
        <center>
        <b>🔬 Model Performance Testing 🔬</b><br>
        Test GPT-2's performance under various conditions<br>
        <i>See how the 2019 model handles stress!</i>
        </center>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Test Configuration")
        st.markdown("---")
        
        test_type = st.selectbox(
            "Test Type",
            ["short", "medium", "long", "repetitive"],
            help="Type of prompts to test with"
        )
        
        num_iterations = st.slider(
            "Number of Iterations",
            min_value=1,
            max_value=10,
            value=3,
            help="How many times to run the test"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Test Types")
        st.markdown("""
        **Short**: Single words or very short prompts
        
        **Medium**: Normal-length questions
        
        **Long**: Extended prompts with context
        
        **Repetitive**: Prompts with repeated patterns
        """)
    
    # Load model
    if "stress_test_model" not in st.session_state or "stress_test_tokenizer" not in st.session_state:
        with st.spinner("⏳ Loading GPT-2 model..."):
            model, tokenizer = load_model()
            st.session_state["stress_test_model"] = model
            st.session_state["stress_test_tokenizer"] = tokenizer
    else:
        model = st.session_state["stress_test_model"]
        tokenizer = st.session_state["stress_test_tokenizer"]
    
    # Test controls
    st.markdown('<div class="test-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        run_button = st.button("🚀 Run Stress Test", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Run test
    if run_button:
        st.markdown("---")
        st.markdown("### 📈 Test Results")
        
        with st.spinner(f"🧪 Running {num_iterations} iterations..."):
            results = run_stress_test(model, tokenizer, test_type, num_iterations)
        
        # Display results
        total_time = sum(r["time"] for r in results)
        success_count = sum(1 for r in results if "Success" in r["status"])
        avg_time = total_time / len(results) if results else 0
        
        # Summary
        st.markdown(f"""
        <div class="success-box">
            <b>Test Complete!</b><br>
            ✅ Successful: {success_count}/{len(results)}<br>
            ⏱️ Total Time: {total_time:.2f}s<br>
            📊 Average Time: {avg_time:.2f}s per iteration
        </div>
        """, unsafe_allow_html=True)
        
        # Individual results
        for result in results:
            status_class = "result-box" if "Success" in result["status"] else "error-box"
            escaped_prompt = html.escape(result["prompt"][:100])
            escaped_response = html.escape(result["response"][:200])
            
            st.markdown(f"""
            <div class="{status_class}">
                <b>Iteration {result["iteration"]}</b> {result["status"]}<br>
                <b>Prompt:</b> {escaped_prompt}<br>
                <b>Response:</b> {escaped_response}<br>
                <b>Time:</b> {result["time"]:.2f}s
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <center>
    <div class="retro-badge">Stress Test v1.0</div>
    <p style="color: #FFFFFF; font-family: monospace; font-size: 1.2em;">
    🔬 GPT-2 Performance Testing Tool 🔬<br>
    <i>Pushing 2019 technology to its limits!</i>
    </p>
    </center>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
