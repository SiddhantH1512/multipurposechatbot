import requests
from config import Config
from schemas.email_schema import EmailExtraction
from utils import ChatGrokModel
from prompts.email_prompts import email_prompt_template
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import json

model = ChatGrokModel()

search = DuckDuckGoSearchRun(region="us-en")
@tool
def email_action_extractor(email_test: str) -> dict:
    '''
    Extracts actionable tasks from an email.
    '''
    model = ChatGrokModel()
    structured_model = model.with_structured_output(EmailExtraction)
    prompt = email_prompt_template.format_messages(
        email_text=email_test
    )
    result: EmailExtraction = structured_model.invoke(prompt)

    return result.model_dump()


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={Config.ALPHA_VANTAGE_API_KEY}"
    r = requests.get(url)
    return r.json()



all_tools = [email_action_extractor, calculator, get_stock_price, search]




if __name__ == "__main__":
    test_email = "Hi Siddhant,Once your account is full, syncing and other features will be paused:Syncing across devicesSaving shared filesCreating new documentsAdding new uploadsBacking up photosEven adding a single new file could pause syncing across your devices and limit your ability to access your files when you need them."
    print(email_action_extractor(test_email))