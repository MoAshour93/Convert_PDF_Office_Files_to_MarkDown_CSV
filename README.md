# 📚 MarkItDown Converter App

> 🔄 A Streamlit-powered application that converts various file formats to Markdown and CSV, with integrated LLM capabilities for Q&A pair generation - perfect for LLM fine-tuning datasets.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 🌟 Features

- 📄 Multi-format file conversion support:
  - PDF documents
  - PowerPoint presentations
  - Word documents
  - Excel spreadsheets
  - Images (EXIF + OCR)
  - Audio (EXIF + transcription)
  - HTML files
  - Text-based formats (CSV, JSON, XML)
  - ZIP archives
- 🤖 LLM integration for enhanced conversion
- ❓ Automatic Q&A pair generation
- 🔤 Embedding generation support
- 📊 CSV export with structured data
- 🎯 Perfect for creating LLM fine-tuning datasets

## 🛠️ Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) (for LLM and embedding features)
- Virtual environment capability

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/MoAshour93/Convert_PDF_Office_Files_to_MarkDown_CSV.git
cd markitdown-converter-app
```

2. Create and activate virtual environment:
```bash
python -m venv markitdown_env
# On Windows:
markitdown_env\Scripts\activate
# On Unix or MacOS:
source markitdown_env/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install Ollama (required for LLM features):
- Visit [Ollama.ai](https://ollama.ai)
- Follow installation instructions for your OS
- Install embedding models using


'''bash
ollama pull {model_name}(e.g. all-minlm , nomic-embed-text ...etc)
'''

## 💫 Usage

1. Start the application:
```bash
streamlit run MarkItDown_Conversion_App_v0.4.py
```

2. Access the web interface:
- Open your browser
- Navigate to `http://localhost:8501`

3. Choose your conversion path:
- **File to Markdown**: Direct file conversion with optional LLM enhancement
- **Markdown to CSV**: Extract or generate Q&A pairs from markdown
- **Direct File to Q&A**: Convert files directly to Q&A pairs with embeddings

## 🎓 Core Package

This application is built on top of Microsoft's [MarkItDown](https://github.com/microsoft/markitdown) utility. Basic Python usage:

```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("test.xlsx")
print(result.text_content)
```

With LLM integration:
```python
from markitdown import MarkItDown
from openai import OpenAI

client = OpenAI()
md = MarkItDown(llm_client=client, llm_model="gpt-4")
result = md.convert("example.jpg")
print(result.text_content)
```

## 📜 License

This project is licensed under CC BY-NC-SA 4.0:
- For research and non-commercial use only
- Attribution required
- Modifications allowed with creator notification
- Commercial use requires explicit permission

## 👤 Author

**Mohamed Ashour**

Connect with me:
- 📧 Email: mo_ashour1@outlook.com
- 💼 LinkedIn: [Mohamed Ashour](https://www.linkedin.com/in/mohamed-ashour-0727/)
- 🌐 Website: [APC Mastery Path](https://www.apcmasterypath.co.uk)
- 📽️Youtube:[APC Mastery Path](https://youtube.com/@APCMasteryPath)

## 🤝 Contributing

Feel free to:
- Open issues
- Submit Pull Requests
- Share improvements
- Report bugs

For major changes, please open an issue first to discuss what you would like to change.

## 🙏 Acknowledgments

- Microsoft's [MarkItDown](https://github.com/microsoft/markitdown) team for the core conversion utility
- [Streamlit](https://streamlit.io/) for the web framework
- [Ollama](https://ollama.ai) for LLM integration capabilities
