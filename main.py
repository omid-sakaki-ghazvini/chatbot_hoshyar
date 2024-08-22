from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import streamlit as st

pathFileForModel = 'document_for_chatBot.txt'

tokenizer = AutoTokenizer.from_pretrained("m3hrdadfi/xlmr-large-qa-fa")
model = AutoModelForQuestionAnswering.from_pretrained("m3hrdadfi/xlmr-large-qa-fa")
pipe = pipeline("question-answering", model=model, tokenizer=tokenizer )

st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    text-align: right;
    color: blue;
}
.large-font {
    font-size:20px !important;
    text-align: right;
    color: blue;
}
.small-font {
    font-size:20px !important;
    text-align: right;
    color: green;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">سلام، من یک هوش مصنوعی هستم که میتوانم به سوال شما در رابطه با کارنامه دکتر قالیباف پاسخ بدهم</p>', unsafe_allow_html=True)

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
    st.markdown('<p class="small-font">سوال شما:</p>', unsafe_allow_html=True)
    st.write(question)
    st.markdown('<p class="large-font">پاسخ سوال شما براساس داده هایی که من به آن دسترسی دارم:</p>', unsafe_allow_html=True)
    response = chatBot(question, context)
    st.write(response)
