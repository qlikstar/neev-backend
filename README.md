## How to Run the service 

1. Go into the repo : `cd neev-backend`
2. Run the requirements: `pip install -r requirements.txt`
3. Run `streamlit run app.py --server.port 8000`

### How to run a different LLM

1. Go into the `app.py`
2. Select the model you want to run by **commenting out** others:
    ```
    model = AnthropicModel(AnthropicModelIdentifier.CLAUDE3_HAIKU, VoyageEmbedIdentifier.VOYAGE_2)
    model = OllamaModel(OllamaModelIdentifier.GEMMA_2B)
    model = OpenAIModel()
    ```
3. Run Streamlit on a different port : `streamlit run app.py --server.port 8001`

### How to run a model locally:

1. Follow the instructions to install Ollama and download the model based on CPU config: https://github.com/ollama/ollama
   ```
   $> curl -fsSL https://ollama.com/install.sh | sh
   $> ollama pull gemma:2b
   ```
2. Now, run the model locally 
3. Then, go into the `app.py`
4. Select the model you want to run by **commenting out** others:
    ```
    model = OllamaModel(OllamaModelIdentifier.GEMMA_2B)
    ```
5. Run Streamlit on a different port : `streamlit run app.py --server.port 8002`