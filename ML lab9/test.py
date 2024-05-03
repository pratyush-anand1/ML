import streamlit as st
import sqlite3
import re
from ply import lex, yacc

conn = sqlite3.connect("student.db")
cursor = conn.cursor()

# Tokens
tokens = (
    'SELECT', 'UPDATE', 'SHOW', 'ALL',
    'IDENTIFIER', 'NUMBER', 'STRING', 'AND', 'LESS_THAN', 'MORE_THAN', 'EQUAL_TO'
)

# Tokens
t_SELECT = r'select'
t_UPDATE = r'update'
t_SHOW = r'show'
t_ALL = r'all'
t_AND = r'and'
t_LESS_THAN = r'less than'
t_MORE_THAN = r'more than'
t_EQUAL_TO = r'equal to'
t_ignore = ' \t\n'



def find_query(sentence):
    select_regex = r"Please give me the (.+) of the student"
    update_regex = r"Please update the (.+) of the student having roll no (\d+) to (\d+)"
    conditional_regex = r"Please show the (.+) of the students whose (.+)"
    all_regex = r"Please give me all the (.+) of the student"

    select_match = re.search(select_regex, sentence)
    update_match = re.search(update_regex, sentence)
    conditional_match = re.search(conditional_regex, sentence)
    all_match = re.search(all_regex, sentence)
    keywords = {
        'more than': '>',
        'less than': '<',
        'equal to': '=', 
        'and': 'AND', 
    }

    if select_match:
        select = select_match.group(1)
        if select == "roll numbers":
            sql_query = "SELECT roll FROM student;"  # Adjusted to match schema
        elif select == "all the information":
            sql_query = "SELECT * FROM student;"
        else:
            sql_query = f"SELECT {select} FROM student;"
    elif update_match:
        col = update_match.group(1)
        roll_no = update_match.group(2)
        value = update_match.group(3)
        sql_query = f"UPDATE student SET {col}={value} WHERE roll={roll_no};"  # Adjusted to match schema
    elif conditional_match:
        select = conditional_match.group(1)
        condition = conditional_match.group(2)
        for x in keywords:
            if x in select: 
                select=select.replace(x,keywords[x])
            if x in condition: 
                condition=condition.replace(x,keywords[x])
        
        sql_query = f"SELECT {select} FROM student WHERE {condition};"
    elif all_match:
        sql_query = "SELECT * FROM student;"
    else:
        sql_query = "Invalid input."

    return sql_query

def execute_query(query):
    try:
        cursor.execute(query)
        conn.commit()
        db = cursor.fetchall()
        return db
    except sqlite3.Error as e:
        conn.rollback()
        return f"Error executing SQL query: {e}"
    
    

st.title('SQL Query Generator')
text = st.text_input("Text to parse")

if st.button('Find'):
    # parsed_result = parser.parse(lexer=lexer)
    result = find_query(text)
    st.markdown('## Results')
    st.markdown('Query: ')
    st.write(result)
    db = execute_query(result)

    st.markdown('Output from database: ')
    if isinstance(db, str):
        st.write(db)
    else:
        st.write(db)