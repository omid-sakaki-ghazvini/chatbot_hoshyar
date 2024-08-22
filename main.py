from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import streamlit as st

pathFileForModel = 'document_for_chatBot.txt'

tokenizer = AutoTokenizer.from_pretrained("m3hrdadfi/xlmr-large-qa-fa")
model = AutoModelForQuestionAnswering.from_pretrained("m3hrdadfi/xlmr-large-qa-fa")
pipe = pipeline("question-answering", model=model, tokenizer=tokenizer )

original_title = '<p style="font-family:Courier; color:Blue; font-size: 20px;">سلام، من یک هوش مصنوعی هستم که میتوانم به سوال شما در رابطه با کارنامه دکتر قالیباف پاسخ بدهم</p>'
st.markdown(original_title, unsafe_allow_html=True)

with open(pathFileForModel, 'r', encoding='utf-8') as file:
    context = file.readlines()
    context = [''.join(context)]
    context = context[0]



question = st.text_area("لطفا سوال خود را بپرسید:")    



def chatBot(question, context):
    kwargs = {}
    r = pipe(question=question, context=context, **kwargs)    
    answer = " ".join([token.strip() for token in r["answer"].strip().split() if token.strip()]) 
    return(answer)  
      
        
if question:
    st.write("<h5>سوال شما :</h5>")
    st.write(question)
    st.write("پاسخ سوال شما براساس داده هایی که من به آن دسترسی دارم:")
    response = chatBot(question, context)
    st.write(response)
