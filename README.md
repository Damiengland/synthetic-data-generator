# Synthetic Data Generator

This project utilizes the Llama 3.1 8B Instruct model to generate high-quality, synthetic datasets based on user-defined prompts. It provides a simple interface using Gradio to interact with the model and save the generated data to Google Drive.

## Features

- **Powerful LLM:** Leverages the `meta-llama/Llama-3.1-8B-Instruct` model for text generation.
- **Efficient Inference:** Uses 4-bit quantization (`bitsandbytes`) to run the model efficiently on consumer-grade GPUs (like Google Colab's T4).
- **Customizable Prompts:** Easily configurable system and user prompts to tailor the data generation to specific needs.
- **JSON Output:** Designed to produce structured JSON output, ideal for creating datasets.
- **Interactive UI:** A simple web interface powered by Gradio for easy interaction.
- **Google Drive Integration:** Automatically saves the generated data to your Google Drive.

## Requirements

The project is designed to be run in a Google Colab environment. The following libraries are required:

- `torch`
- `torchvision`
- `torchaudio`
- `requests`
- `bitsandbytes`
- `transformers`
- `accelerate`
- `openai`
- `gradio`
- `huggingface_hub`

## Setup & Installation

1.  **Clone the Repository:**
    Clone or download the project files to your local machine or Google Drive.

2.  **Open in Google Colab:**
    Upload and open the `synthetic_data_generator.ipynb` notebook in Google Colab.

3.  **API Keys and Tokens:**
    This project requires access to the Hugging Face model and the OpenAI API. You need to store your credentials as secrets in Google Colab.
    - Go to the "Secrets" tab in the left sidebar of your Colab notebook.
    - Add the following secrets:
        - `HF_TOKEN`: Your Hugging Face access token with read permissions.
        - `OPENAI_API_KEY`: Your OpenAI API key.

4.  **Install Dependencies:**
    Run the first code cell in the notebook to install all the required Python packages.

    ```python
    !pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
    !pip install -q requests bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0 openai
    ```

5.  **Authorize Google Drive:**
    When you run the cell to mount Google Drive, you will be prompted to authorize Colab to access your Drive files. This is necessary for saving the output.

## How to Run

1.  **Execute Cells:** Run the cells in the notebook sequentially from top to bottom.
2.  **Wait for Model to Load:** The model is several gigabytes and will take some time to download and load into the GPU memory.
3.  **Launch the UI:** The final cell will launch a Gradio interface directly within the notebook output.
4.  **Generate Data:**
    - Enter your prompt in the input text box (e.g., `I need 5 user reviews for a new coffee machine.`).
    - Click the "Submit" button.
    - The generated text will appear in the output box.
5.  **Find Your Data:** The generated output will also be saved as a `.txt` file in your Google Drive, located at `/MyDrive/04_Projects/Python Projects/synthetic_data_generator/Llama_Responses/`.

## Author

Damien England

## License

This project is licensed under the MIT License.