import pandas as pd
import numpy as np
import random
import string
import streamlit as st
import re
import ast

def main():
    """
    Основная функция для создания интерфейса генератора упражнений по английскому языку.

    Процесс работы функции:
    1. Отображает заголовок страницы.
    2. Добавляет элементы управления для загрузки файла CSV и указания количества заданий.
    3. При загрузке файла CSV и выборе количества заданий, генерирует упражнения и предоставляет пользователю интерфейс
       для взаимодействия с ними.
    4. Проверяет ответы пользователя и отображает результаты.

    Функция не принимает никаких параметров, так как все данные вводятся пользователем через элементы управления.
    """
    st.title("Генератор упражнений по английскому языку")

    # Добавьте элемент управления для загрузки файла CSV
    uploaded_file = st.file_uploader("Загрузить файл CSV", type="csv")

    # Добавьте элемент управления для указания количества заданий
    num_questions = st.number_input("Укажите количество заданий", min_value=1, step=1)

    # Проверьте, был ли файл загружен
    if uploaded_file is not None:
        # Прочитайте файл CSV в DataFrame
        df = pd.read_csv(uploaded_file)
        
        df['answers'] = df['answers'].apply(ast.literal_eval)
        
        correct_count = 0
        total_count = 0
    
        # Ограничение количества заданий
        if num_questions > len(df):
            num_questions = len(df)
        
        # Генерируем задания на основе данных DataFrame
        for index, row in df.iterrows():
            if total_count >= num_questions:
                break
            
            types = row['type']
            object_word = row['object']
            answers = row['answers']

            st.subheader(f"Задание {index+1}")

            if types == 'missing_word':
                sentence = row['raw']
                code_word_index = eval(row['object'])[1]  # Get the code_word_index
                words = sentence.split()
                
                words[code_word_index] = "_____"
                sentence_with_blank = " ".join(words)
    
                st.write(f"Найдите пропущенное слово в предложении:")
                st.write(sentence_with_blank)
                user_answer = st.selectbox("Выберите правильное слово:", answers, key=f"answer_{index}")
                # Проверка ответа пользователя
                if user_answer == '----':
                    st.write("")
                elif user_answer == eval(object_word)[0]:
                    st.write('<span style="color:green">Верно!</span>', unsafe_allow_html=True)
                    correct_count += 1
                else:
                    st.write('<span style="color:red">Неверно!</span>', unsafe_allow_html=True)

            elif types == 'sentence_wrong':
                user_answer = st.selectbox("Выберите правильное предложение:", answers, key=f"answer_{index}")
                # Проверка ответа пользователя
                if user_answer == '----':
                    st.write("")
                elif user_answer == row['raw']:
                    st.write('<span style="color:green">Верно!</span>', unsafe_allow_html=True)
                    correct_count += 1
                else:
                    st.write('<span style="color:red">Неверно!</span>', unsafe_allow_html=True)

            elif types == 'sentence_structure':
                st.write(f"Определите члены предложения:")
                sentence = row['raw']
                object_dict = object_word
                st.write(sentence)
                answers = dict(answers)
                for key, value in answers.items():
                    user_input = st.text_input(f"{key}", key=f"input_{index}_{key}")
                    # Проверка ответа пользователя
                    if user_input == '':
                        st.write("")
                    elif user_input.lower() == value.lower():
                        st.write(f"{key} - <span style='color:green'>Верно!</span>", unsafe_allow_html=True)
                        correct_count += 1
                    else:
                        st.write(f"{key} - <span style='color:red'>Неверно!</span>", unsafe_allow_html=True)

            else:  # fill_gaps
                sentence = row['raw']
                code_word_index = eval(row['object'])[1]  # Get the code_word_index
                words = sentence.split()
                
                words[code_word_index] = "_____"
                sentence_with_blank = " ".join(words)
    
                st.write(f"Заполните пропуск в предложении:")
                st.write(sentence_with_blank)
                user_answer = st.text_input("Введите правильное слово:", key=f"input_{index}")
                # Check user's answer
                if user_answer == '':
                    st.write("")
                elif user_answer.lower() == eval(object_word)[0].lower():
                    st.write('<span style="color:green">Верно!</span>', unsafe_allow_html=True)
                    correct_count += 1
                else:
                    st.write('<span style="color:red">Неверно!</span>', unsafe_allow_html=True)

            st.write("---")
            total_count += 1

        # Подсчет правильных заданий
        st.write(f"Правильных заданий: {correct_count}/{total_count}")


if __name__ == "__main__":
    main()
