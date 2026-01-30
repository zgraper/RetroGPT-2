# RetroGPT-2 💾

A Streamlit chatbot that lets users converse with GPT-2 inside a late-90s / early-2000s web interface, explicitly highlighting how far language models have progressed.

## 🌟 Features

- **Retro 90s/2000s Aesthetic**: Experience the nostalgia with Windows 95/98 inspired UI, complete with classic colors, borders, and fonts
- **GPT-2 Chatbot**: Chat with the 2019 GPT-2 model (117M parameters)
- **Customizable Parameters**: Adjust temperature, top-p, and response length
- **Conversation History**: Maintains chat context for more coherent conversations
- **Educational**: Learn about how far language models have evolved since GPT-2

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/zgraper/RetroGPT-2.git
cd RetroGPT-2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 🎨 Retro Design Elements

- **Windows 95/98 Color Scheme**: Classic teal background with silver window frames
- **Monospace Fonts**: Courier New and system monospace fonts for authentic old-school web feel
- **Outset/Inset Borders**: Classic 3D button effects from the 90s
- **Box Shadows**: Depth effects mimicking old UI frameworks
- **Blinking Title**: Just like those geocities websites!

## 🤖 About GPT-2

GPT-2 was released by OpenAI in February 2019. At the time, it was considered remarkably powerful with its 117M parameters (in the base model). This chatbot uses the original GPT-2 model to showcase:

- How language models have evolved
- The rapid progress in AI capabilities
- The foundations that led to modern models like ChatGPT

**Key Differences from Modern Models:**
- No reinforcement learning from human feedback (RLHF)
- No instruction tuning
- Pure next-token prediction
- Much smaller scale (117M vs billions of parameters)

## 📋 Requirements

- Python 3.8+
- streamlit
- transformers
- torch

## 🛠️ Configuration

Adjust model parameters in the sidebar:
- **Response Length**: 50-200 tokens
- **Temperature**: 0.1-2.0 (controls creativity)
- **Top P**: 0.1-1.0 (nucleus sampling parameter)

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT-2
- Hugging Face for the transformers library
- Streamlit for the amazing framework
