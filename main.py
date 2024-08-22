from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import streamlit as st

pathFileForModel = 'document_for_chatBot.txt'

pipeline = pipeline(task= 'question-answering', model="m3hrdadfi/xlmr-large-qa-fa", tokenizer="m3hrdadfi/xlmr-large-qa-fa")

st.title("سلام، من یک هوش مصنوعی هستم که میتوانم به سوال شما در رابطه با کارنامه دکتر قالیباف پاسخ بدهم")

if loadData :
    with open(pathFileForModel, 'r', encoding='utf-8') as file:
        context = file.readlines()
        context = [''.join(context)]
        print(">> data load")
        context = context[0]



question = st.text_area("لطفا سوال خود را بپرسید:")    



def chatBot(question, context):
    kwargs = {}
    r = pipeline(question=question, context=context, **kwargs)    
    answer = " ".join([token.strip() for token in r["answer"].strip().split() if token.strip()]) 
    return(answer)  
      
        
if question:
    st.write("سوال شما:")
    st.write(question)
    st.write("پاسخ سوال شما براساس داده هایی که من به آن دسترسی دارم:")
    response = chatBot(question, context)
    st.write(response)
