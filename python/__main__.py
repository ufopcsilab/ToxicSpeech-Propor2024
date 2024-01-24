#!/usr/bin/python
# -*- encoding: utf-8 -*-

#!pip install simpletransformers

import pandas as pd
import requests
import argparse
import time
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
import numpy as np
import random
import torch
from scipy.special import softmax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

def main(seed_value=42,
         train_file='',
         test_file='',
         train_data_column=0,
         train_label_column=1,
         test_data_column=0,
         test_label_column=1,
         embedding_dimension=16,
         model_args_num_train_epochs=3,
         model_args_train_batch_size=8,
         model_args_eval_batch_size=8,
         model_args_overwrite_output_dir=True,
         model_args_save_steps=-1,
         model_args_save_model_every_epoch=False,
         model_args_learning_rate=3e-5,
         model_args_fp16=True,
         using_model='bert',
         pre_trainned_model='',
         DEFAULT_API_KEY = '',
         URL_API = "",
         ):
    
    if pre_trainned_model == 'neuralmind/bert-large-portuguese-cased' or pre_trainned_model == 'adalbertojunior/distilbert-portuguese-cased' or pre_trainned_model == 'pierreguillou/bert-large-cased-squad-v1.1-portuguese':
        # Set the seed for PyTorch
        torch.manual_seed(seed_value)

        # If you're using CUDA or cuDNN, also set the seed for those
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Set the seed for NumPy
        np.random.seed(seed_value)

        # Set the seed for Python's random module
        random.seed(seed_value)


        # Load training and test data
        arquivo_csv_treino = train_file
        arquivo_csv_teste = test_file

        dados_treino = pd.read_csv(arquivo_csv_treino)
        dados_teste = pd.read_csv(arquivo_csv_teste)

        train_df = pd.DataFrame({'text': dados_treino.iloc[:, train_data_column], 'labels': dados_treino.iloc[:, train_label_column]})
        eval_df = pd.DataFrame({'text': dados_teste.iloc[:, test_data_column], 'labels': dados_teste.iloc[:, test_label_column]})

        print(train_df.head())

        print(eval_df['labels'].unique())
        print(eval_df['labels'].unique())

        # Create the ranking model
        model_args = {
            'num_train_epochs': model_args_num_train_epochs,
            'train_batch_size': model_args_train_batch_size,
            'eval_batch_size': model_args_eval_batch_size,
            'overwrite_output_dir': model_args_overwrite_output_dir,
            'save_steps': model_args_save_steps,
            'save_model_every_epoch': model_args_save_model_every_epoch,
            'learning_rate': model_args_learning_rate,
            'fp16': model_args_fp16,
        }

        model = ClassificationModel(
            using_model,
            pre_trainned_model,
            num_labels=2,
            args=model_args,
            use_cuda=True,  # If using a GPU, you can change this to True
        )

        # Train the model on training data
        model.train_model(train_df)

        # Evaluate the model on test data
        predictions, raw_outputs = model.predict(eval_df['text'].tolist())

        # Convert Raw_outputs to Probabilities Using the SoftMax function
        probabilities = np.exp(raw_outputs) / np.sum(np.exp(raw_outputs), axis=1, keepdims=True)
        predicted_probs = probabilities[:, 1]  # Probabilities for Class 1 (toxic)

       # Add 'predicts' and 'probabilities' to the evaluation dataframe
        eval_df['predictions'] = predictions
        eval_df['probabilities'] = predicted_probs

       # Save updated dataframe in a CSV file
        eval_df.to_csv('./results/Bertimbau_test_datasetName.csv', index=False)

    elif pre_trainned_model == "PORTULAN/albertina-ptbr":
        # Check if CUDA is available and set the default device to cuda
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Sets the first available GPU as default

        # Model and tokenizer
        model_name = pre_trainned_model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # Move the model to the GPU
        if torch.cuda.is_available():
            model = model.to("cuda")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Upload your dataset
        dataset = load_dataset('csv', data_files={
            'train': train_file,
            'test': test_file
        })

        # Tokenization function for the "text" column
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

        # Apply the tokenization function to the dataset
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Adding 'Toxic' column as 'labels'
        tokenized_datasets["train"] = tokenized_datasets["train"].add_column("labels", tokenized_datasets["train"]["Toxic"])
        tokenized_datasets["test"] = tokenized_datasets["test"].add_column("labels", tokenized_datasets["test"]["Toxic"])

        # Training Settings
        training_args = TrainingArguments(
            output_dir="results/albertina",
            evaluation_strategy="epoch",
            per_device_train_batch_size=model_args_train_batch_size,
            per_device_eval_batch_size=model_args_eval_batch_size,
            num_train_epochs=model_args_num_train_epochs,
            save_strategy="epoch",
            logging_dir='./logs',
        )

        # Set the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
        )

        # To train
        trainer.train()

        # Predictions
        predictions, raw_outputs = trainer.predict(tokenized_datasets["test"])

        # Convert raw_outputs to probabilities using the softmax function
        probabilities = np.exp(raw_outputs) / np.sum(np.exp(raw_outputs), axis=1, keepdims=True)
        predicted_probs = probabilities[:, 1]  # Odds for class 1 (toxic)

        # Converting tokenized_datasets["test"] to DataFrame and adding 'predictions' and 'probabilities'
        eval_df = pd.DataFrame(tokenized_datasets["test"])
        eval_df['predictions'] = predictions
        eval_df['probabilities'] = predicted_probs

        # Save the updated DataFrame to a CSV file
        eval_df.to_csv('./results/albertina.csv', index=False)
    
    elif pre_trainned_model == "maritalk":
        HEADERS = { "authorization": f"Key {DEFAULT_API_KEY}"}
        dados_treinamento = pd.read_csv(train_file)
        dados_teste = pd.read_csv(test_file)
        URL = URL_API

        def create_prompt(texto, method, n_instances):
            if method == "zero-shot":
                return ("Você está realizando análise de texto. O texto a seguir foi utilizado em um contexto de uma rede social. "
                        "Responda apenas 'sim' ou 'não' caso o texto abaixo contenha discurso de ódio, ofensa, agressividade ou outras formas de toxicidade: " + texto)
            else:
                exemplos_toxicos = dados_treinamento[dados_treinamento['Toxic'] == 1.0].sample(n_instances//2, random_state=seed_value)
                exemplos_nao_toxicos = dados_treinamento[dados_treinamento['Toxic'] == 0.0].sample(n_instances//2, random_state=seed_value)

                messages = []
                for _, row in exemplos_toxicos.iterrows():
                    messages.append({"role": "user", "content": "Você está realizando análise de texto. O texto a seguir foi utilizado em um contexto de uma rede social. Responda apenas 'sim' ou 'não' caso o texto abaixo contenha discurso de ódio, ofensa, agressividade ou outras formas de toxicidade: " + row['clean_text']})
                    messages.append({"role": "assistant", "content": "sim, é tóxico."})

                for _, row in exemplos_nao_toxicos.iterrows():
                    messages.append({"role": "user", "content": "Você está realizando análise de texto. O texto a seguir foi utilizado em um contexto de uma rede social. Responda apenas 'sim' ou 'não' caso o texto abaixo contenha discurso de ódio, ofensa, agressividade ou outras formas de toxicidade: "+ row['clean_text']})
                    messages.append({"role": "assistant", "content": "não é tóxico."})

                messages.append({"role": "user", "content": "Você está realizando análise de texto. O texto a seguir foi utilizado em um contexto de uma rede social. Responda apenas 'sim' ou 'não' caso o texto abaixo contenha discurso de ódio, ofensa, agressividade ou outras formas de toxicidade: " + texto})
                return messages
            
        def get_response(text, headers, method="zero-shot", n_instances=50):
            prompt = create_prompt(text, method, n_instances)
            data = {
                "messages": prompt,
                "do_sample": False,
                "temperature": 0,
                "model": pre_trainned_model,
                "repetition_penalty": 1.2
            }

            max_retries = 5

            for i in range(max_retries):
                resposta = requests.post(URL, headers=headers, json=data)

                if resposta.status_code == 200:
                    # Depending on the structure of the answer, this may need tweaking.
                    return resposta.json()['answer']

                elif resposta.status_code == 429:  # Rate limited
                    sleep_time = min(5.0, 0.2 * (2 ** i))  # Exponential backoff starting from 200ms, max 5 seconds
                    print(f"Rate limited! Tentando novamente em {sleep_time} segundos...")
                    time.sleep(sleep_time)

                else:
                    print(f"Erro na requisição: {resposta.status_code} - {resposta.text}")
                    return None

            # If the code reaches this far, it means that all attempts have failed.
            print("Número máximo de tentativas alcançado!")
            return None
        
        def main(args):
            HEADERS['Authorization'] = f'Key {args.api_key}'
            texts = dados_teste['clean_text'].tolist()

            results = []
            for text in texts:
                print(text)
                result = get_response(text, HEADERS, method=args.method, n_instances=args.n_instances)
                if result:
                    print(result + '\n')
                    results.append(result)
                else:
                    print("Failed to get a result!\n")

                time.sleep(6)  # Delay to respect API's rate limit

            # Save the results
            output_filename = f"./results/Maritaca_datasetName_{args.method}_{args.n_instances if args.method == 'few-shot' else 'zero-shot'}.csv"
            
            
            dados_teste['predictions'] = results
            dados_teste.to_csv(output_filename, index=False)
            print("Classificação concluída e resultados salvos.")

        #python seu_script.py --method few-shot --n_instances 100 --api_key YOUR_KEY

        parser = argparse.ArgumentParser(description='Send texts to OpenAI API for classification.')
        parser.add_argument('--method', type=str, choices=['zero-shot', 'few-shot'], default='zero-shot', help='Learning method to use.')
        parser.add_argument('--n_instances', type=int, default=50, help='Number of instances to be used for few-shot learning.')
        parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY, help='OpenAI API key.')

        args = parser.parse_args()
        main(args)

    elif pre_trainned_model == "chatGPT_35turbo_Few_shot_Prompt1":

        # API OPENAI configuration
        KEY = DEFAULT_API_KEY
        URL = URL_API  # Updated for the correct endpoint
        HEADERS = {
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json"
        }

        # Reading the data sets
        arquivo_treinamento = train_file
        arquivo_teste = test_file
        dados_treinamento = pd.read_csv(arquivo_treinamento)
        dados_teste = pd.read_csv(arquivo_teste)

       # Selection of EXAMPLES FEW-SHOT
        SEMENTE = seed_value
        exemplos_toxicos = dados_treinamento[dados_treinamento['Toxic'] == 1.0].sample(10, random_state=SEMENTE)
        exemplos_nao_toxicos = dados_treinamento[dados_treinamento['Toxic'] == 0.0].sample(10, random_state=SEMENTE)

        # Creation of initial messages
        messages = [{"role": "system", "content": "Você está realizando análise de texto. O texto a seguir foi utilizado em um contexto de uma rede social. Responda apenas 'sim' ou 'não' caso você considere esse texto tóxico."}]
        for _, row in exemplos_toxicos.iterrows():
            messages.append({"role": "user", "content": f"Texto: {row['clean_text']}"})
            messages.append({"role": "assistant", "content": "sim"})

        for _, row in exemplos_nao_toxicos.iterrows():
            messages.append({"role": "user", "content": f"Texto: {row['clean_text']}"})
            messages.append({"role": "assistant", "content": "não"})

       # Function for classifying text with ulot and retry with exponential backoff
        def classificar_texto(texto, max_retries=5):
            chat_messages = messages + [{"role": "user", "content": f"Texto: {texto}"}]
            data = {
                "model": "gpt-3.5-turbo",
                "messages": chat_messages
            }

            for i in range(max_retries):
                resposta = requests.post(URL, headers=HEADERS, json=data)
                if resposta.status_code == 200:
                    return resposta.json()['choices'][0]['message']['content'].strip()
                elif resposta.status_code == 429:  # Rate limited
                    sleep_time = 2 ** (i + 1)  # Exponential backoff: 2, 4, 8, 16, ...
                    print(f"Rate limited! Tentando novamente em {sleep_time} segundos...")
                    time.sleep(sleep_time)
                else:
                    print(f"Erro na requisição: {resposta.status_code} - {resposta.text}")
                    sleep_time = 2 ** (i + 1)
                    print(f"Esperando {sleep_time} segundos antes da próxima tentativa...")
                    time.sleep(sleep_time)

            raise Exception("Número máximo de tentativas alcançado!")

        # Classifying each tweet using unce-shot
        resultados = []
        for tweet in dados_teste['clean_text']:
            resultado = classificar_texto(tweet)
            print(tweet)
            print(resultado + '\n')
            resultados.append(resultado)

        # Save results
        dados_teste['predictions'] = resultados
        dados_teste.to_csv('./results/Chatgpt_35turbo_datasetName_prompt_1_fewshot.csv', index=False)
        print("Completed classification and saved results.")

    elif pre_trainned_model == "chatGPT_35turbo_Few_shot_Prompt2":
        # API OPENAI configuration
        KEY = DEFAULT_API_KEY
        URL = URL_API  # Updated for the correct endpoint
        HEADERS = {
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json"
        }

        # Reading the data sets
        arquivo_treinamento = train_file
        arquivo_teste = test_file
        dados_treinamento = pd.read_csv(arquivo_treinamento)
        dados_teste = pd.read_csv(arquivo_teste)

        # Selection of EXAMPLES FEW-SHOT
        SEMENTE = seed_value
        exemplos_toxicos = dados_treinamento[dados_treinamento['Toxic'] == 1.0].sample(10, random_state=SEMENTE)
        exemplos_nao_toxicos = dados_treinamento[dados_treinamento['Toxic'] == 0.0].sample(10, random_state=SEMENTE)

        # Creation of initial messages
        messages = [{"role": "system", "content": "Você está realizando análise de texto."}]
        for _, row in exemplos_toxicos.iterrows():
            messages.append({"role": "user", "content": f"O texto a seguir foi utilizado em um contexto de uma rede social. Responda apenas 'sim' ou 'não' caso o texto abaixo contenha discurso de ódio, ofensa, agressividade ou outras formas de toxicidade: {row['clean_text']}"})
            messages.append({"role": "assistant", "content": "sim"})

        for _, row in exemplos_nao_toxicos.iterrows():
            messages.append({"role": "user", "content": f"O texto a seguir foi utilizado em um contexto de uma rede social. Responda apenas 'sim' ou 'não' caso o texto abaixo contenha discurso de ódio, ofensa, agressividade ou outras formas de toxicidade: {row['clean_text']}"})
            messages.append({"role": "assistant", "content": "não"})

        # Function for classifying text with ulot and retry with exponential backoff
        def classificar_texto(texto, max_retries=5):
            chat_messages = messages + [{"role": "user", "content": f"O texto a seguir foi utilizado em um contexto de uma rede social. Responda apenas 'sim' ou 'não' caso o texto abaixo contenha discurso de ódio, ofensa, agressividade ou outras formas de toxicidade: {texto}"}]
            data = {
                "model": "gpt-3.5-turbo",
                "messages": chat_messages
            }
            
            for i in range(max_retries):
                resposta = requests.post(URL, headers=HEADERS, json=data)
                if resposta.status_code == 200:
                    return resposta.json()['choices'][0]['message']['content'].strip()
                elif resposta.status_code == 429:  # Rate limited
                    sleep_time = 2 ** (i + 1)  # Exponential backoff: 2, 4, 8, 16, ...
                    print(f"Rate limited! Tentando novamente em {sleep_time} segundos...")
                    time.sleep(sleep_time)
                else:
                    print(f"Erro na requisição: {resposta.status_code} - {resposta.text}")
                    sleep_time = 2 ** (i + 1)
                    print(f"Esperando {sleep_time} segundos antes da próxima tentativa...")
                    time.sleep(sleep_time)

            raise Exception("Número máximo de tentativas alcançado!")

        # Classifying each tweet using unce-shot
        resultados = []
        for tweet in dados_teste['clean_text']:
            resultado = classificar_texto(tweet)
            print(tweet)
            print(resultado + '\n')
            resultados.append(resultado)

        # Save results
        dados_teste['predictions'] = resultados
        dados_teste.to_csv('./results/Chatgpt_35turbo_datasetName_prompt_2_fewshot.csv', index=False)
        print("Completed classification and saved results.")


    elif pre_trainned_model == "chatGPT_35turbo_Zero_shot_Prompt2":
        # API OPENAI configuration
        KEY = DEFAULT_API_KEY
        URL = URL_API
        HEADERS = {
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json"
        }

        # Reading the data sets
        arquivo_treinamento = train_file
        arquivo_teste = test_file
        dados_teste = pd.read_csv(arquivo_teste)

        # Initial message
        messages = [{"role": "system", "content": "Você está realizando análise de texto."}]

        # Function for classifying text in zero-shot and retry mode with exponential backoff
        def classificar_texto(texto, max_retries=5):
            chat_messages = messages + [{"role": "user", "content": f"O texto a seguir foi utilizado em um contexto de uma rede social. Responda apenas 'sim' ou 'não' caso o texto abaixo contenha discurso de ódio, ofensa, agressividade ou outras formas de toxicidade: {texto}"}]
            data = {
                "model": "gpt-3.5-turbo",
                "messages": chat_messages
            }
        
            for i in range(max_retries):
                resposta = requests.post(URL, headers=HEADERS, json=data)
                if resposta.status_code == 200:
                    return resposta.json()['choices'][0]['message']['content'].strip()
                elif resposta.status_code == 429:  # Rate limited
                    sleep_time = 2 ** (i + 1)  # Exponential backoff: 2, 4, 8, 16, ...
                    print(f"Rate limited! Tentando novamente em {sleep_time} segundos...")
                    time.sleep(sleep_time)
                else:
                    print(f"Erro na requisição: {resposta.status_code} - {resposta.text}")
                    sleep_time = 2 ** (i + 1)
                    print(f"Esperando {sleep_time} segundos antes da próxima tentativa...")
                    time.sleep(sleep_time)

            raise Exception("Número máximo de tentativas alcançado!")

        # Sort each tweet using zero-shot
        resultados = []
        for tweet in dados_teste['clean_text']:
            resultado = classificar_texto(tweet)
            print(tweet)
            print(resultado + '\n')
            resultados.append(resultado)

        # Save
        dados_teste['predictions'] = resultados
        dados_teste.to_csv('./results/Chatgpt_35turbo_datasetName_prompt_2_zeroshot.csv', index=False)
        print("Completed classification and saved results.")

    else:
        print("Invalid technique.")
        return

if __name__ == "__main__":
    main(seed_value=42,
         train_file='../datas/train.csv',
         test_file='../datas/test.csv',
         train_data_column=0,
         train_label_column=1,
         test_data_column=0,
         test_label_column=1,
         embedding_dimension=16,
         model_args_num_train_epochs=3,
         model_args_train_batch_size=8,
         model_args_eval_batch_size=8,
         model_args_overwrite_output_dir=True,
         model_args_save_steps=-1,
         model_args_save_model_every_epoch=False,
         model_args_learning_rate=3e-5,
         model_args_fp16=True,
         using_model='bert',
         pre_trainned_model='neuralmind/bert-large-portuguese-cased',
         DEFAULT_API_KEY = 'your_api_key',
         URL_API = "url_of_api_selected"
    )
