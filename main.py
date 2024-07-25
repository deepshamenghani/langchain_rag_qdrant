from langchain_openai import ChatOpenAI
import os
import dotenv

dotenv.load_dotenv()
llmclient = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API"))

while True:
    humaninput = input(">> ")
    result = humaninput    
    print(result)