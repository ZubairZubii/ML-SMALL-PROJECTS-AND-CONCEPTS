from dotenv import load_dotenv
import google.generativeai as genai
import os

def get_gemini_response(recipy_message):
    
   # ... (rest of your `get_gemini_response` function)

    # Load environment variables
    load_dotenv()



    genai.configure(api_key= "AIzaSyDs1ntkqMQv_1mrECtqspt2CCS9WjxLp_Y")

    model = genai.GenerativeModel(model_name="gemini-pro")
    # Craft a well-structured prompt template
    prompt_template = f"""Based on the information you provided ({recipy_message}), here is a recipe for you:

    ## Title:

    ## Ingredients:

    1. 
    2. 
    ...

    ## Instructions:

    1. 
    2. 
    ...
    """


    try:
        # Generate the response
        response = model.generate_content(prompt_template)
        
        print(response.text)

        # Process the response for display (optional)
        return response.text.strip()

    except Exception as e:
        print(f"Error generating response: {e}")
        return "There was an error generating the recipe. Please try again later."