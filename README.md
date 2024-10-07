# Customize Small Language Models on AWS with Automotive terminology

In this repository, we will go through the steps required for fine-tuning foundation models on Amazon SageMaker, by using an open-source dataset from Hugging Face for code diagnostic for the Automotive domain, 
deploy the model in a SageMaker Real-time inference endpoint or in Amazon Bedrock, and evaluate it on Amazon SageMaker Studio.

You can run this repository from Amazon SageMaker Studio.

## Prerequistes

The notebooks are currently using the latest [Hugging Face](https://github.com/aws/deep-learning-containers/blob/master/available_images.md) Training Container available for the region `us-east-1`. If you are running the notebooks in a different region, make sure to update the *ImageUri* in the file [config.yaml](./config.yaml).

## Dataset

[sp01/Automotive_NER](https://huggingface.co/datasets/sp01/Automotive_NER)

## Notebooks

1. [01_llama-3-8b-qlora-sft](./01_llama-3-8b-qlora-sft.ipynb): In this notebook, we are extracting a subset of the original dataset by using as distribution mechanism the TfidfVectorizer, by considering the importance of a word in a document (row), based on its frequency in the document and its rarity across the entire corpus of documents (rows)
2. [02_llama-3-8b-deployment-sagemaker](./02_llama-3-8b-deployment-sagemaker.ipynb): In this notebook, we are going to deploy the fine-tuned model in Amazon SageMaker, and perform evaluation with base LLMs in Amazon Bedrock
3. [03_llama-3-8b-deployment-bedrock.ipynb](./03_llama-3-8b-deployment-bedrock.ipynb): In this notebook, we are going to deploy the fine-tuned model in Amazon Bedrock with Custom Import Models, and perform evaluation with base LLMs in Amazon Bedrock

## Step-by-step guidance

### Data analysis and processing on Amazon SageMaker Studio

Use Amazon SageMaker Studio with JupyterLab app for data anlysis and prepration of the dataset, by extracting a relevant subset 
from it. Use remote compute capacity by running SageMaker Training jobs using @remote function

### Model fine-tuning

Prepare prompt templates and run fine-tuning jobs using SageMaker Training and @remote funciton from Amazon SageMaker Studio

### Model Deployment

Deploy the fine-tuned Large Language Model on Amazon SageMaker or import the model in Amazon Bedrock with Custom Import Models

### Model Evaluation

The model evaluation is performed by deploying the fine-tuned model in Amazon SageMaker or Amazon Bedrock, and by using the base model in Amazon Bedrock.
Evaluation metrics identified are BLEU Score (for text quality) and Normalized Levenshtein distance (for reponse accuracy)