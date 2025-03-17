# WEBWHISPER - Talk to Websites! 🚀

## Introduction
WEBWHISPER is a Streamlit-based chatbot that allows users to interact with any website by pasting its URL. Using web scraping, FAISS vector storage, and OpenAI's language model, the app retrieves relevant content from the website and provides AI-powered responses. Additionally, it offers text-to-speech (TTS) functionality to convert AI responses into audio.

## Features
- 🌐 **Interact with Websites** - Enter a URL to scrape and index the content.
- 🔍 **AI-Powered Responses** - Uses Retrieval-Augmented Generation (RAG) to provide accurate answers.
- 🗂 **FAISS Vector Storage** - Efficiently indexes and retrieves website content.
- 🎤 **Text-to-Speech (TTS)** - Converts AI responses into speech.
- 🔄 **Context-Aware Chat** - Keeps track of conversation history for better responses.

## Installation

### Clone the Repository
```sh
git clone https://github.com/yourusername/webwhisper.git
cd webwhisper
```

### Install Dependencies
Create a virtual environment (optional but recommended):
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install required packages:
```sh
pip install -r requirements.txt
```

### Set Up Environment Variables
Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Application
Start the Streamlit app by running:
```sh
streamlit run app.py
```

## Usage
1. **Enter a website URL** in the sidebar.
2. The app will **scrape and index the content**.
3. **Ask questions** in the chat input.
4. AI will respond based on website content.
5. Click **Play** to hear the response as audio.

## Live Demo
Try the app here: [WEBWHISPER Live](https://webwhisper.streamlit.app/)

## Technologies Used
- **Streamlit** - UI Framework
- **LangChain** - AI-powered retrieval and chat processing
- **FAISS** - Vector storage for efficient content retrieval
- **OpenAI API** - LLM for intelligent responses
- **gTTS** - Text-to-Speech conversion
- **Web Scraping** - Extracting data from websites

## Contributing
Pull requests are welcome! Feel free to fork and submit improvements.

## License
This project is licensed under the MIT License.

## Contact
For questions or feedback, reach out at [your email].
