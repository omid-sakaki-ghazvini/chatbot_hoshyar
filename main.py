from langchain_community.document_loaders import PyPDFium2Loader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import streamlit as st

pdfFilePath = 'dr_ghalibaf.pdf'
pathFileForModel = 'document_for_chatBot.txt'

pipeline = pipeline(task= 'question-answering', model="m3hrdadfi/xlmr-large-qa-fa", tokenizer="m3hrdadfi/xlmr-large-qa-fa")

st.title("سلام، من یک هوش مصنوعی هستم که میتوانم به سوال شما در رابطه با کارنامه دکتر قالیباف پاسخ بدهم")

def dataLoad(pdfFilePath):
    loader = PyPDFium2Loader(pdfFilePath) 
    data = loader.load()
    context = []
    for i in range(len(data)):
        context.append(data[i].page_content)
    file = open(pathFileForModel , mode='w', encoding='utf-8')    
    for j in range(len(context)):
      file.writelines(context[j])  
    file.close()
    loadData = False 
    return()


if loadData :
    dataLoad(pdfFilePath)
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
