import pandas as pd
import numpy as np
import random
import string
import nltk
from nltk import sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
import spacy
import pyinflect
import streamlit as st
import re

corpus = api.load('text8')

model = Word2Vec(corpus)
nlp = spacy.load("C:/Users/denis/AppData/Roaming/Python/Python310/site-packages/en_core_web_md/en_core_web_md-3.5.0")

class SynonymsGenerator:
    
    def __init__(self, model, nlp):
        self.model = model
        self.nlp = nlp
    
    def generate_object(self, sentence):
        words = sentence.split()
        words = [word.strip(string.punctuation) for word in words] 
        words = [word for word in words if word] 
        for _ in range(100):
            code_word = np.random.choice(words)
            code_word.lower()    
            if code_word in model.wv.key_to_index:
                break
   
        code_word_index = words.index(code_word)
   
        return [code_word, code_word_index]

    def generate_synonyms(self, code_word):
        code_word = code_word[0].lower()
        if code_word in ['is', 'was', 'were']:
            synonym_1 = 'is'
            synonym_2 = 'was'
            synonym_3 = 'were'
        elif code_word in ['a', 'the', 'an']:
            synonym_1 = 'a'
            synonym_2 = 'the'
            synonym_3 = 'an'
        else:
            similar_words = self.model.wv.most_similar(code_word, topn=5)
            choices = random.sample(similar_words, 2)
            synonym_1 = choices[0][0]
            synonym_2 = choices[1][0]
            synonym_3 = code_word
            
        answer_choices = [synonym_1, synonym_2, synonym_3]
        random.shuffle(answer_choices)
        answer_choices.insert(0, '----')
        return answer_choices
                                                                
    def generate_incorrect_sentence(self, sentence):
        
        sentences = set()

        while len(sentences) < 5:
            doc = nlp(sentence)
            tokens = [token for token in doc] 
            if not tokens:
                continue
        
            sentence = sentence[:1].lower() + sentence[1:]

            random_token = random.choice(tokens)
            
            while random_token.is_punct:  # Check if the randomly chosen token is punctuation
                random_token = random.choice(tokens)

            if random_token.tag_ in ['JJ', 'JJR', 'JJS']:
                new_word = random_token._.inflect(random.sample(['JJ', 'JJR', 'JJS'], k=1)[0])
            elif random_token.tag_ in ['RB', 'RBR', 'RBS']:
                new_word = random_token._.inflect(random.sample(['RB', 'RBR', 'RBS'], k=1)[0])
            elif random_token.tag_ in ['NN', 'NNS']:
                new_word = random_token._.inflect(random.sample(['NN', 'NNS'], k=1)[0])
            elif random_token.tag_ in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                new_word = random_token._.inflect(random.sample(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], k=1)[0])
            elif random_token.tag_ == 'TO':
                new_word = ''
            elif random_token.tag_ == 'DT' and random_token.text.lower() != 'this':
                new_word = random_token._.inflect(random.sample(['DT'], k=1)[0])
            else:
                continue

            if new_word:
                tokens = [t.text if t != random_token else new_word for t in tokens]
            else:
                tokens = [t.text for t in tokens if t != random_token]

            new_sentence = ' '.join(tokens)

            # Capitalize the sentence again
            new_sentence = new_sentence[:1].upper() + new_sentence[1:]
            sentence = sentence[:1].upper() + sentence[1:]
            
            new_sentence = new_sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')                   .replace(' ;', ';').replace(' :', ':').replace(' "', '"').replace(" '", "'")                   .replace(' -', '-')

            if new_sentence != sentence:
                sentences.add(new_sentence)

        sentences = list(sentences)

        first_value = random.choice(sentences)
        sentences.remove(first_value)
        second_value = random.choice(sentences)
        
        answer_choices = [first_value, second_value, sentence]
        random.shuffle(answer_choices)
        answer_choices.insert(0, '----')
        
        
        return answer_choices
    
    def generate_sentence_structure(self, sentence):
    
        # Обработка предложения
        doc = nlp(sentence)
    
        # Извлечение членов предложения
        subject = None
        predicate = None
        objects = None
    
        for token in doc:
            if "subj" in token.dep_:
                subject = token.text
                break  # Прекратить поиск после первого найденного подлежащего
    
        # Поиск сказуемого
        for token in doc:
            if token.head == token and token.pos_ == "VERB":
                predicate = token.text
                break
    
        # Извлечение дополнений
        for token in doc:
            if "obj" in token.dep_:
                objects = token.text
                break
    
        parts_sentence = [subject, predicate, objects]

        answers = {'Subject': parts_sentence[0], 'Predicate': parts_sentence[1], 'Object': parts_sentence[2]}
        answers = {key: value for key, value in answers.items() if value is not None}
    
        return answers

with open('C:/Users/denis/Desktop/Little_Red_Cap_ Jacob_and_Wilhelm_Grimm.txt', 'r') as file:
    text = file.read()
sentences = sent_tokenize(text)

def generate_exercises(text_input):
    
    sentences = sent_tokenize(text)

    # Создаем DataFrame с столбцом 'raw'
    df = pd.DataFrame({'raw': sentences})

    # Добавляем столбец 'word_count' с количеством слов
    df['word_count'] = df['raw'].apply(lambda x: len(x.split()))

    # Фильтруем предложения с количеством слов >= 4
    df = df[df['word_count'] >= 4].drop('word_count', axis=1)
    df = df.reset_index(drop=True)

    # Типы заданий
    task_types = ['missing_word', 'sentence_structure', 'sentence_wrong', 'fill_gaps']

    # Добавляем столбец 'type' со случайно выбранными значениями
    np.random.seed(42)
    probabilities = [0.25, 0.25, 0.25, 0.25]
    df['type'] = np.random.choice(task_types, len(df), p=probabilities)

    # Создаем экземпляр класса SynonymsGenerator
    generator = SynonymsGenerator(model, nlp)

    df['object'] = None
    df['answers'] = None

    for index, row in df.iterrows():
        types = row['type']
        if types == 'missing_word':
            df.at[index, 'object'] = generator.generate_object(row['raw'])
            df.at[index, 'answers'] = generator.generate_synonyms(df.at[index, 'object'])
        elif types == 'sentence_wrong':
            df.at[index, 'object'] = row['raw']
            df.at[index, 'answers'] = generator.generate_incorrect_sentence(row['raw'])
        elif types == 'sentence_structure':
            df.at[index, 'object'] = generator.generate_sentence_structure(row['raw']).values()
            df.at[index, 'answers'] = generator.generate_sentence_structure(row['raw'])
        else:
            df.at[index, 'object'] = generator.generate_object(row['raw'])
            df.at[index, 'answers'] = generator.generate_object(row['raw'])
    return df

def main():
    st.title("Генератор упражнений по английскому языку")

    text_input = st.text_area("Введите текст", "")

    if text_input:  # Проверка, является ли переменная text_input пустой
        if st.button("Сгенерировать упражнения"):
            # Generate exercises
            df = generate_exercises(text_input)

            csv_data = df.to_csv(index=False)
            st.download_button(
                "Скачать файл",
                data=csv_data,
                file_name='english_exercises.csv',
                mime='text/csv'
            )

if __name__ == '__main__':
    main()
