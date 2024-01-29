# Document-Interactive-Chat
Simple Chainlit app to have interaction with your documents. This Chainlit app only accepts .txt and .pdf file types. This is because a PDF ingestion pipeline has been implemented with PyPDFLoader. You can also add different ingestor pipelines or create your own ones for your preferred files.

### Chat with your documents 🚀
- [Huggingface](https://huggingface.co/) to load an embedding model to embed words
- [LangChain](https://python.langchain.com/docs/get_started/introduction.html) as a Framework for LLM pipelines
- [Chainlit](https://docs.chainlit.io/overview) for deploying UI chat interface
- [Ollama](https://github.com/ollama/ollama) for LLM model loading
- [Pinecone](https://www.pinecone.io/) as vector database and vector similarity querying

## System Requirements

It is recommended to have Python 3.10.5 installed. However, lower versions have not been tried.

---

## Steps to Replicate

1. Fork this repository and create a codespace in GitHub OR Clone it locally.
   ```
   git clone CHANGE
   cd CHANGE
   ```

2. Install Ollama by following the instructions at https://ollama.ai/. Then, install the model to use by running the following command in the terminal:
   ```
   ollama run "model"
   ```
   In this case the model used is mistral:instruct. However, this can be changed in the following code line in main.py:
   ```
   Ollama(model="mistral:instruct",)
   ```
   Embbeding model can also be changed in
   ```
   embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
   ```

3. Set up a virtual environment with pyenv. First, make sure you have pyenv installed. Then run the following commands:
   ```
   pyenv install 3.10.5
   pyenv virtualenv 3.10.5 .venv
   pyenv local .venv
   ```

4. Run the following command in the terminal to install necessary python packages:
   ```
   pip install -r requirements.txt
   ```
   Note that some additional dependencies might be needed. If you want to easily install these additional dependencies, you can use a `requirements2.txt` file (if available) by running:
   ```
   pip install -r requirements2.txt
   ```

5. Update the `.env` file with your personal API keys:

   Open the `example.env` file in your project directory and input your personal API keys as follows:
   ```
   CHAINLIT_AUTH_SECRET=your_chainlit_auth_secret
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENV=your_pinecone_env
   ```
   Replace `chainlit_auth_secret`, `pinecone_api_key`, and `pinecone_env` with your actual API keys for Chainlit and Pinecone. Save the changes to the `.env` file.
   Rename `example.env` to `.env` with the following command:
   ```
   cp example.env .env
   ```

6. Create a Pinecone account and an index, then update the `index_name` variable in the code with your index name.

7. Run the following command in your terminal to run the app UI:
   ```
   chainlit run main.py -w
   ```

8. The utility of authentication is available with the username `admin` and password `admin123`. This is implemented with a .csv. That is because its purpose is to provide a simple demonstration of the capability to implement an authentication system with Chainlit.