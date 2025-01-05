"""
MarkItDown Conversion App v0.4
------------------------------
A Streamlit application for converting various file formats to Markdown, CSV, and Q&A pairs.
Supports multiple file uploads, LLM integration, embeddings, and flexible output options.

Main Required Dependencies:
- streamlit
- pandas
- markitdown
- requests

To create this application:
1. Install required packages:
   pip install streamlit pandas markitdown requests

2. Set up Ollama (required for LLM and embedding features):
   - Install Ollama from https://ollama.ai
   - Run Ollama server locally
   - Install desired models including embedding models

3. Run the application:
   streamlit run app.py
"""
# =============================================
#Step 0: Importing necessary Libraries
# =============================================

import streamlit as st
import os
import pandas as pd
from markitdown import MarkItDown
import requests
import json
import re
from json.decoder import JSONDecodeError
from typing import List, Tuple, Optional, Dict, Any

# =============================================
# STEP 1: OLLAMA INTEGRATION
# =============================================

def get_ollama_models() -> Tuple[List[str], List[str]]:
    """
    Fetch available models from local Ollama server and separate them into LLM and embedding models.
    
    Returns:
        tuple: (llm_models, embedding_models) where each is a list of model names
    """
    embedding_models = ["mxbai-embed-large:latest", "nomic-embed-text:latest","all-minilm:latest"]
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            all_models = [model['name'] for model in response.json()['models']]
            # Filter out embedding models from general LLM models
            llm_models = [model for model in all_models if model not in embedding_models]
            # Filter available embedding models
            available_embedding_models = [model for model in embedding_models if model in all_models]
            return llm_models, available_embedding_models
        return [], []
    except:
        return [], []

def setup_ollama_client(model_name: str):
    """
    Create an Ollama client wrapper for API interactions.
    
    Args:
        model_name (str): Name of the Ollama model to use
    
    Returns:
        OllamaClient: Instance of the client wrapper
    """
    class OllamaClient:
        def __init__(self, model: str):
            self.model = model
            
        def chat_completion_create(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
            """
            Send chat completion request to Ollama with proper response handling
            
            Args:
                messages: List of message dictionaries with 'role' and 'content'
            
            Returns:
                dict: Parsed response from Ollama
            """
            url = 'http://localhost:11434/api/chat'
            data = {
                'model': self.model,
                'messages': messages,
                'stream': False
            }
            
            try:
                response = requests.post(url, json=data)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    try:
                        return response.json()
                    except JSONDecodeError as e:
                        lines = response.text.strip().split('\n')
                        if lines:
                            for line in reversed(lines):
                                try:
                                    return json.loads(line)
                                except JSONDecodeError:
                                    continue
                            raise ValueError(f"Could not parse response as JSON: {str(e)}")
                else:
                    raise ValueError(f"Unexpected content type: {content_type}")
                
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Request to Ollama failed: {str(e)}")
        
        def create_embeddings(self, text: str, embedding_model: str) -> List[float]:
            """
            Generate text embeddings using specified embedding model
            
            Args:
                text: Input text to generate embeddings for
                embedding_model: Name of the embedding model to use
            
            Returns:
                list: Vector embeddings
            """
            url = 'http://localhost:11434/api/embeddings'
            data = {
                'model': embedding_model,
                'prompt': text
            }
            
            try:
                response = requests.post(url, json=data)
                response.raise_for_status()
                return response.json().get('embedding', [])
            except (requests.exceptions.RequestException, JSONDecodeError) as e:
                st.warning(f"Failed to generate embeddings: {str(e)}")
                return []

    return OllamaClient(model_name)

# =============================================
# STEP 2: FILE PROCESSING FUNCTIONS
# =============================================

def process_single_file(
    file, 
    use_llm: bool = False, 
    ollama_client: Optional[Any] = None, 
    selected_model: Optional[str] = None
) -> Tuple[str, str]:
    """
    Process a single file and convert it to markdown.
    
    Args:
        file: Uploaded file object
        use_llm: Whether to use LLM enhancement
        ollama_client: Optional Ollama client instance
        selected_model: Optional selected model name
    
    Returns:
        tuple: (filename, markdown_content)
    """
    temp_path = f"temp_{file.name}"
    try:
        with open(temp_path, 'wb') as f:
            f.write(file.getbuffer())
        
        if use_llm and ollama_client and selected_model:
            md = MarkItDown(llm_client=ollama_client, llm_model=selected_model)
        else:
            md = MarkItDown()
        
        result = md.convert(temp_path)
        return file.name, result.text_content
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def extract_qa_pairs_from_markdown(markdown_text: str) -> List[List[str]]:
    """
    Extract question-answer pairs from markdown text with improved pattern matching.
    
    Args:
        markdown_text: Input markdown text
    
    Returns:
        list: List of [question, answer] pairs
    """
    qa_pairs = []
    lines = markdown_text.split('\n')
    current_question = None
    current_answer = None
    
    for line in lines:
        line = line.strip().replace('**', '')
        
        if not line:
            continue
            
        question_match = re.match(r'^Question:\s*(.+)', line, re.IGNORECASE)
        if question_match:
            if current_question is not None and current_answer is not None:
                qa_pairs.append([current_question, current_answer])
            
            current_question = question_match.group(1).strip()
            current_answer = None
            continue
            
        answer_match = re.match(r'^Answer:\s*(.+)', line, re.IGNORECASE)
        if answer_match and current_question is not None:
            current_answer = answer_match.group(1).strip()
    
    if current_question is not None and current_answer is not None:
        qa_pairs.append([current_question, current_answer])
    
    return qa_pairs

def generate_qa_pairs_with_llm(
    content: str, 
    ollama_client: Any,
    chunk_size: int = 4000
) -> List[List[str]]:
    """
    Generate Q&A pairs from content using LLM with improved chunking and formatting.
    
    Args:
        content: Input text content
        ollama_client: Configured Ollama client instance
        chunk_size: Maximum size of text chunks to process
    
    Returns:
        list: List of [question, answer] pairs
    """
    def chunk_text(text: str, size: int, overlap: int = 1000) -> List[str]:
        """
        Split text into chunks of approximately equal size with specified overlap
        
        Args:
            text: Input text to be chunked
            size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
        
        Returns:
            list: List of text chunks with overlap
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, word in enumerate(words):
            word_size = len(word) + 1  # Add 1 for space
            current_chunk.append(word)
            current_size += word_size
            
            # Check if current chunk exceeds size limit
            if current_size > size:
                chunks.append(' '.join(current_chunk))
                
                # Calculate how many words to keep for overlap
                overlap_size = 0
                overlap_words = []
                for w in reversed(current_chunk):
                    if overlap_size + len(w) + 1 > overlap:
                        break
                    overlap_words.insert(0, w)
                    overlap_size += len(w) + 1
                
                # Start new chunk with overlap words
                current_chunk = overlap_words
                current_size = overlap_size
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    all_qa_pairs = []
    chunks = chunk_text(content, chunk_size)
    
    for i, chunk in enumerate(chunks):
        try:
            prompt = f"""
            Generate a useful number of questions and answer pairs from the following text.
            Make sure that the questions are intact and not a mere repition.
            Make sure that the answers are explanatory based on the text provided but at the same time not excessively lengthy.
            Make questions specific and factual.
            Format each pair exactly as:
            Question: <question>
            Answer: <answer>

            Use single newline between question and answer, double newline between pairs.

            Text chunk {i+1}/{len(chunks)}:
            {chunk}
            """
            
            response = ollama_client.chat_completion_create([
                {
                    "role": "system", 
                    "content": """"You are a precise questions and answers pair generator that creates factual, specific questions and answers from text. 
                    You are an experienced Quantity Surveyor and you have a degree in civil engineering.
                    You work in an esteemed consultancy in the United Kingdom that serves clients allover the world.
                    You are well aware of the different RICS Guidance notes and the latest advancements in the construction industry.
                    You lead a group of senior quantity surveyors working on a variety of projects including residential, commercial, infrastructure, transport, rail and much more.
                    You want to build a database which explore the different areas of competence of the RICS alongside their corresponding levels and how to provide the best answer possible for every level in every competency.
                    """
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ])
            
            if isinstance(response, dict):
                generated_text = None
                
                if 'message' in response and isinstance(response['message'], dict):
                    generated_text = response['message'].get('content')
                elif 'response' in response:
                    generated_text = response['response']
                elif 'content' in response:
                    generated_text = response['content']
                
                if not generated_text:
                    raise ValueError("No content found in LLM response")
                
                chunk_qa_pairs = extract_qa_pairs_from_markdown(generated_text)
                all_qa_pairs.extend(chunk_qa_pairs)
                
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
            
        except Exception as e:
            st.error(f"Error processing chunk {i+1}: {str(e)}")
            continue
    
    return all_qa_pairs

def process_file_to_qa(
    file,
    ollama_client: Any,
    embedding_model: str
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Process a file directly to QA pairs with embeddings.
    
    Args:
        file: Uploaded file object
        ollama_client: Configured Ollama client instance
        embedding_model: Name of the embedding model to use
    
    Returns:
        tuple: (qa_pairs, embeddings) where qa_pairs is a list of [question, answer] 
               and embeddings is a list of embedding vectors
    """
    # First convert file to markdown
    filename, content = process_single_file(file, use_llm=True, ollama_client=ollama_client)
    
    # Generate QA pairs
    qa_pairs = generate_qa_pairs_with_llm(content, ollama_client)
    
    # Generate embeddings for each QA pair
    embeddings = []
    for question, answer in qa_pairs:
        # Combine question and answer for embedding
        combined_text = f"Question: {question}\nAnswer: {answer}"
        embedding = ollama_client.create_embeddings(combined_text, embedding_model)
        embeddings.append(embedding)
    
    return qa_pairs, embeddings

def create_qa_dataframe(
    qa_pairs: List[List[str]], 
    embeddings: List[List[float]]
) -> pd.DataFrame:
    """
    Create a DataFrame from QA pairs and their embeddings.
    
    Args:
        qa_pairs: List of [question, answer] pairs
        embeddings: List of embedding vectors
    
    Returns:
        pd.DataFrame: DataFrame containing QA pairs and embeddings
    """
    # Create base DataFrame with ID, Question, and Answer columns
    df = pd.DataFrame(qa_pairs, columns=['Question', 'Answer'])
    df.insert(0, 'ID', range(1, len(df) + 1))
    
    # Add embedding columns
    if embeddings and len(embeddings[0]) > 0:
        embedding_cols = [f'embedding_{i}' for i in range(len(embeddings[0]))]
        embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
        df = pd.concat([df, embedding_df], axis=1)
    
    return df

# =============================================
# STEP 3: MAIN APPLICATION
# =============================================

def main():
    """Main application entry point with enhanced functionality"""
    
    st.title("File Converter to Markdown, CSV, and Q&A Generation")
    st.subheader("Created by Mohamed Ashour, Founder of APC Mastery Path.")
    
    # --------- Sidebar Configuration ---------
    with st.sidebar:
        st.header("About This App")
        st.markdown("""
        ## Purpose
        This application serves as a versatile document conversion tool, specifically designed to:
        - Convert various file formats to Markdown
        - Extract or generate Question-Answer pairs
        - Generate embeddings for Q&A pairs
        - Export structured content to CSV format
        
        ## Main Features
        - **Multi-format Support**: Convert PDFs, Office documents, images, audio, and more
        - **Intelligent Conversion**: LLM integration for enhanced content understanding
        - **Q&A Generation**: Multiple methods:
          1. Direct file to Q&A conversion
          2. Markdown intermediary conversion
          3. Extract existing Q&A pairs
        - **Embedding Support**: Generate embeddings using specialized models
        - **CSV Export**: Structured output with optional embeddings
        
        ## How to Use
        1. **File to Markdown**:
           - Upload files
           - Choose LLM enhancement options
           - Convert and download
        
        2. **Markdown to CSV**:
           - Convert markdown with existing Q&A pairs
           - Generate new Q&A pairs using LLM
        
        3. **Direct File to Q&A**:
           - Upload files
           - Select LLM and embedding models
           - Get Q&A pairs with embeddings
        
        ## Creator Contact
        **Mohamed Ashour**  
        
        💼 [LinkedIn](https://www.linkedin.com/in/mohamed-ashour-0727/)  
        
        📧 Email:  
        - mohamed.ashour@apcmasterypath.co.uk  
        - mo_ashour1@outlook.com
        
        ## License
        © 2024 Mohamed Ashour. All rights reserved.
        
        This application is licensed under CC BY-NC-SA 4.0:
        - For research and non-commercial use only
        - Attribution required
        - Modifications and improvements are allowed
        - Modified versions must be shared with the creator
        - Commercial use requires explicit permission from the creator
        
        Contact the creator for commercial licensing inquiries.
        
        """)
        st.divider()
        st.caption("Version 0.4")
    
    # --------- Main Tabs ---------
    tab1, tab2, tab3 = st.tabs([
        "File to Markdown", 
        "Markdown to CSV", 
        "Direct File to Q&A"
    ])
    
    # Get available models once for all tabs
    llm_models, embedding_models = get_ollama_models()
    
    # Tab 1: File to Markdown Conversion
    with tab1:
        st.subheader("Convert Files to Markdown")
        
        uploaded_files = st.file_uploader(
            "Choose files to convert", 
            type=['pdf', 'pptx', 'docx', 'xlsx', 'jpg', 'jpeg', 'png', 'mp3', 
                  'wav', 'html', 'csv', 'json', 'xml', 'zip'],
            accept_multiple_files=True,
            key="tab1_uploader"
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} files for conversion")
            
            output_format = st.radio(
                "Choose output format:",
                ["Combined Markdown (Single File)", "Separate Markdown Files"],
                key="tab1_format"
            )
            
            use_llm = st.checkbox("Use LLM for enhanced conversion", key="tab1_llm")
            selected_model = None
            ollama_client = None
            
            if use_llm:
                if llm_models:
                    selected_model = st.selectbox(
                        "Select LLM Model", 
                        llm_models,
                        key="tab1_model"
                    )
                    ollama_client = setup_ollama_client(selected_model)
                else:
                    st.warning("No LLM models found. Please check Ollama installation.")
            
            if st.button("Convert to Markdown", key="tab1_convert"):
                with st.spinner("Converting files..."):
                    try:
                        results = []
                        for file in uploaded_files:
                            filename, content = process_single_file(
                                file, 
                                use_llm, 
                                ollama_client, 
                                selected_model
                            )
                            results.append((filename, content))
                        
                        if output_format == "Combined Markdown (Single File)":
                            combined_content = "\n\n".join([
                                f"# Content from {filename}\n\n{content}" 
                                for filename, content in results
                            ])
                            
                            st.subheader("Combined Conversion Result:")
                            st.text_area(
                                "Markdown Output", 
                                combined_content, 
                                height=300,
                                key="tab1_combined_output"
                            )
                            
                            st.download_button(
                                label="Download Combined Markdown",
                                data=combined_content,
                                file_name="combined_output.md",
                                mime="text/markdown",
                                key="tab1_combined_download"
                            )
                        
                        else:  # Separate files
                            st.subheader("Individual Conversion Results:")
                            for filename, content in results:
                                with st.expander(f"Preview: {filename}"):
                                    st.text_area(
                                        "Markdown Output", 
                                        content, 
                                        height=200,
                                        key=f"tab1_output_{filename}"
                                    )
                                    
                                    st.download_button(
                                        label=f"Download {filename}",
                                        data=content,
                                        file_name=f"{os.path.splitext(filename)[0]}.md",
                                        mime="text/markdown",
                                        key=f"tab1_download_{filename}"
                                    )
                        
                        st.success("Conversion completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error during conversion: {str(e)}")
                        
# Tab 2: Markdown to CSV Conversion
    with tab2:
        st.subheader("Convert Markdown to CSV")
        
        qa_option = st.radio(
            "Choose Q&A extraction method:",
            ["Upload markdown with existing Q&A pairs", "Generate Q&A pairs using LLM"],
            key="tab2_option"
        )
        
        # Add embedding model selection
        selected_embedding_model = None
        selected_llm = None
        
        if qa_option == "Generate Q&A pairs using LLM":
            col1, col2 = st.columns(2)
            
            with col1:
                if llm_models:  # llm_models already excludes embedding models
                    selected_llm = st.selectbox(
                        "Select LLM Model",
                        llm_models,
                        help="Choose the model for Q&A generation",
                        key="tab2_llm"
                    )
                    if selected_llm:
                        st.info(f"Using {selected_llm} for Q&A generation")
                else:
                    st.error("No LLM models found. Please check Ollama installation.")
            
            with col2:
                if embedding_models:
                    selected_embedding_model = st.selectbox(
                        "Select Embedding Model",
                        embedding_models,
                        help="Choose the model for generating embeddings",
                        key="tab2_embedding"
                    )
                else:
                    st.error("No embedding models found. Please install mxbai-embed-large or nomic-embed-text.")
        
        uploaded_markdown = st.file_uploader(
            "Upload Markdown file", 
            type=['md'],
            key="tab2_uploader"
        )
        
        if uploaded_markdown:
            markdown_content = uploaded_markdown.getvalue().decode()
            
            if st.button("Convert to CSV", key="tab2_convert"):
                with st.spinner("Processing markdown content..."):
                    try:
                        if qa_option == "Upload markdown with existing Q&A pairs":
                            qa_pairs = extract_qa_pairs_from_markdown(markdown_content)
                            # Create basic DataFrame without embeddings
                            df = pd.DataFrame(qa_pairs, columns=['Question', 'Answer'])
                            df.insert(0, 'ID', range(1, len(df) + 1))
                        else:
                            if not selected_llm:
                                st.error("Please select an LLM model first")
                                return
                            
                            if not selected_embedding_model:
                                st.error("Please select an embedding model first")
                                return
                            
                            ollama_client = setup_ollama_client(selected_llm)
                            qa_pairs = generate_qa_pairs_with_llm(markdown_content, ollama_client)
                            
                            # Generate embeddings for QA pairs
                            embeddings = []
                            for question, answer in qa_pairs:
                                combined_text = f"Question: {question}\nAnswer: {answer}"
                                embedding = ollama_client.create_embeddings(combined_text, selected_embedding_model)
                                embeddings.append(embedding)
                            
                            # Create DataFrame with embeddings
                            df = create_qa_dataframe(qa_pairs, embeddings)
                        
                        if qa_pairs:
                            st.success(f"Generated {len(qa_pairs)} Q&A pairs!")
                            
                            # Display preview without embedding columns
                            st.subheader("Preview (Q&A Pairs Only):")
                            st.dataframe(df[['ID', 'Question', 'Answer']])
                            
                            # Provide download options
                            st.subheader("Download Options:")
                            
                            # Basic CSV (without embeddings)
                            basic_csv = df[['ID', 'Question', 'Answer']].to_csv(index=False)
                            st.download_button(
                                label="Download Basic CSV (Q&A Only)",
                                data=basic_csv,
                                file_name="qa_pairs.csv",
                                mime="text/csv",
                                key="tab2_basic_download"
                            )
                            
                            # If embeddings exist, provide full CSV download
                            if 'embedding_0' in df.columns:
                                full_csv = df.to_csv(index=False)
                                st.download_button(
                                    label="Download Full CSV (Including Embeddings)",
                                    data=full_csv,
                                    file_name="qa_pairs_with_embeddings.csv",
                                    mime="text/csv",
                                    key="tab2_full_download"
                                )
                        else:
                            st.warning("No question-answer pairs found in the content.")
                            
                    except Exception as e:
                        st.error("Error during conversion process:")
                        st.error(str(e))
    
    # Tab 3: Direct File to Q&A Conversion
    with tab3:
        st.subheader("Convert Files Directly to Q&A Pairs")
        
        # Model Selection
        if not llm_models:
            st.error("No LLM models found. Please check Ollama installation.")
            return
            
        if not embedding_models:
            st.error("No embedding models found. Please install mxbai-embed-large or nomic-embed-text.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_llm = st.selectbox(
                "Select LLM Model for Q&A Generation",
                llm_models,
                help="Choose the model for generating questions and answers",
                key="tab3_llm"
            )
        
        with col2:
            selected_embedding = st.selectbox(
                "Select Embedding Model",
                embedding_models,
                help="Choose the model for generating embeddings",
                key="tab3_embedding"
            )
        
        # File Upload
        uploaded_files = st.file_uploader(
            "Choose files to convert", 
            type=['pdf', 'pptx', 'docx', 'xlsx', 'jpg', 'jpeg', 'png', 'mp3', 
                  'wav', 'html', 'csv', 'json', 'xml', 'zip'],
            accept_multiple_files=True,
            key="tab3_uploader"
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} files for conversion")
            
            if st.button("Generate Q&A Pairs", key="tab3_convert"):
                with st.spinner("Processing files and generating Q&A pairs..."):
                    try:
                        ollama_client = setup_ollama_client(selected_llm)
                        all_qa_pairs = []
                        all_embeddings = []
                        
                        # Process each file
                        for file in uploaded_files:
                            with st.status(f"Processing {file.name}..."):
                                qa_pairs, embeddings = process_file_to_qa(
                                    file,
                                    ollama_client,
                                    selected_embedding
                                )
                                all_qa_pairs.extend(qa_pairs)
                                all_embeddings.extend(embeddings)
                                
                                st.write(f"Generated {len(qa_pairs)} Q&A pairs")
                        
                        if all_qa_pairs:
                            # Create DataFrame with embeddings
                            df = create_qa_dataframe(all_qa_pairs, all_embeddings)
                            
                            st.success(f"Successfully generated {len(all_qa_pairs)} total Q&A pairs!")
                            
                            # Display preview without embedding columns
                            st.subheader("Preview (Q&A Pairs Only):")
                            st.dataframe(df[['ID', 'Question', 'Answer']])
                            
                            # Provide download options
                            st.subheader("Download Options:")
                            
                            # Basic CSV (without embeddings)
                            basic_csv = df[['ID', 'Question', 'Answer']].to_csv(index=False)
                            st.download_button(
                                label="Download Basic CSV (Q&A Only)",
                                data=basic_csv,
                                file_name="qa_pairs.csv",
                                mime="text/csv",
                                key="tab3_basic_download"
                            )
                            
                            # Full CSV (with embeddings)
                            full_csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Full CSV (Including Embeddings)",
                                data=full_csv,
                                file_name="qa_pairs_with_embeddings.csv",
                                mime="text/csv",
                                key="tab3_full_download"
                            )
                            
                        else:
                            st.warning("No Q&A pairs were generated from the files.")
                            
                    except Exception as e:
                        st.error("Error during processing:")
                        st.error(str(e))
                        st.info("""
                        Troubleshooting tips:
                        1. Check if Ollama is running
                        2. Verify the selected models are installed
                        3. Try processing smaller files first
                        4. Check the file formats are supported
                        """)

if __name__ == "__main__":
    main()                        