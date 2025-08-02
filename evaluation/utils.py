import pandas as pd

def get_evaluation_prompt() -> str:
    return """
    Answer this question only with the final number, without any other text. Lower case and upper case letters are considered different characters.
    \n\nQuestion: {question}\n
    """

def generate_prompt(test_df: pd.DataFrame, idx: int) -> str:
    prompt = get_evaluation_prompt()
    prompt = prompt.format(question=test_df.iloc[idx]['question'])
    return prompt

def extract_response(response_message: str) -> int:
    try:
        # Extract the final number from the response and convert to int
        predicted = int(response_message.strip())
        return predicted
    except Exception as e:
        print(f"Unexpected error evaluating response: {str(e)}, {response_message}")
        return None