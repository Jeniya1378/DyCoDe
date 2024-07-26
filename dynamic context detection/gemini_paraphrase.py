import google.generativeai as genai
api_key = "AIzaSyCXJksoxdm8EsjKaHWtzzW2dgMO_lDcKTU"
genai.configure(api_key=api_key)



model = genai.GenerativeModel('gemini-pro')



def get_response(prompt):
    response = model.generate_content(prompt)
    return response.text

def generate_prompt(text, count):
    initial_text = "paraphrase"
    times = "times"
    prompt = f'{initial_text} {count} {times} {text}'
    # print(get_response(prompt))
    paraphrased_text = get_response(prompt)
    return paraphrased_text