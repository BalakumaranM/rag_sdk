
# RAG SDK

A modular RAG SDK built from scratch.

## Verification with Docker (Recommended)

1. **Build the Docker image**:
   ```bash
   docker build -t rag_sdk .
   ```

2. **Run Unit Tests**:
   ```bash
   docker run --rm rag_sdk pytest tests/unit/
   ```

3. **Run the Example Script**:
   Ensure your API keys are set in your local environment, then run:
   ```bash
   docker run --rm \
     -e OPENAI_API_KEY=$OPENAI_API_KEY \
     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
     -e COHERE_API_KEY=$COHERE_API_KEY \
     -e GEMINI_API_KEY=$GEMINI_API_KEY \
     -e VOYAGE_API_KEY=$VOYAGE_API_KEY \
     rag_sdk python main.py
   ```

## Local Development (Optional)

If you prefer running locally without Docker:
1. Install in editable mode: `pip install -e .`
2. Run `python main.py`.
