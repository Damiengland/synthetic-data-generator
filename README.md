# **LLM Joke Generation Exercise**

This project is an educational exercise, a rebuild of a previous project, designed to demonstrate how to perform inference with various large language models (LLMs) from Hugging Face, utilizing 4-bit quantization for efficient memory usage, particularly on GPU-constrained environments like Google Colab. The primary goal is to generate light-hearted jokes using different instruction-tuned models.

## **Features**

* **Hugging Face Model Integration**: Easily load and utilize popular instruction-tuned LLMs from the Hugging Face Hub.  
* **4-bit Quantization**: Implement bitsandbytes for efficient model loading and inference, significantly reducing GPU memory footprint.  
* **Text Generation**: Generate creative and light-hearted text responses based on a given prompt.  
* **Memory Management**: Includes explicit garbage collection and CUDA cache clearing for optimized resource usage.

## **Prerequisites**

To run this notebook, you will need:

* **Google Colab Environment**: The notebook is designed to run in Google Colab, leveraging its GPU capabilities.  
* **Hugging Face Token**: A Hugging Face user access token with read access to models. This token should be stored securely in Colab's user data secrets.

## **Usage**

Follow the steps below to execute the notebook and generate jokes:

1. **Set up Hugging Face Token**:  
   * In Google Colab, go to the left sidebar, click on the "Secrets" (key icon).  
   * Add a new secret with the name HF\_TOKEN and paste your Hugging Face user access token as the value.  
   * Enable "Notebook access" for this secret.  
   * Run the cell that logs into Hugging Face:  
     from google.colab import userdata  
     from huggingface\_hub import login  
     hf\_token \= userdata.get('HF\_TOKEN')  
     login(hf\_token, add\_to\_git\_credential=True)

2. **Define Models**: The notebook pre-defines several model paths. You can choose which model to use for generation.  
   LLAMA \= "meta-llama/Meta-Llama-3.1-8B-Instruct"  
   PHI3 \= "microsoft/Phi-3-mini-4k-instruct"  
   GEMMA2 \= "google/gemma-2-2b-it"  
   QWEN2 \= "Qwen/Qwen2-7B-Instruct"

   *Note*: In the provided notebook, LLAMA is set as the default model for the initial tokenizer and model loading.  
3. **Define Messages**: Customize the messages array to change the prompt. The example is set to generate a light-hearted joke for "Nuke Artists".  
   messages \= \[  
       {"role": "system", "content": "You are a helpful assistant"},  
       {"role": "user", "content": "Tell a light-hearted joke for a room of Nuke Artists"}  
   \]

4. **Run the generate function**: The core logic is encapsulated in the generate function. Call it with your chosen model and messages.  
   \# Example: generate(LLAMA, messages)  
   generate(QWEN2, messages)

   *Ensure the quant\_config is defined before running generate.*

## **Models Supported**

This exercise is configured to work with the following instruction-tuned models from Hugging Face:

* meta-llama/Meta-Llama-3.1-8B-Instruct (assigned to LLAMA)  
* microsoft/Phi-3-mini-4k-instruct (assigned to PHI3)  
* google/gemma-2-2b-it (assigned to GEMMA2)  
* Qwen/Qwen2-7B-Instruct (assigned to QWEN2)

You can modify the generate function call to experiment with different models.

## **Notes & Troubleshooting**

* **Educational Rebuild**: This project is a rebuild of a previous exercise. Its purpose is educational, focusing on the practical aspects of LLM inference with quantization.  
* **Sequential Execution**: Ensure you run the cells in the llm-hf-joker.ipynb notebook sequentially from top to bottom. Skipping cells or running them out of order might lead to NameError or other issues (e.g., model or QWEN2 not being defined).  
* **GPU Availability**: Make sure your Colab runtime type is set to use a GPU (Runtime \-\> Change runtime type \-\> GPU).  
* **Hugging Face Access**: If you encounter issues with model loading, double-check your HF\_TOKEN secret and ensure it has proper access.

## **License**

This project is open-sourced under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for more details.