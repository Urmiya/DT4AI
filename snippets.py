def get_chatgpt_summary(query):
    """
    Function to get summaries from the ChatGPT API using GPT-4.
    
    Web Crawler or news API still needs to be integrated. 
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="gpt-4",
            max_tokens=150
        )
        
        # Accessing the first message content if available
        # Using proper attribute access for Pydantic models
        if chat_completion.choices and chat_completion.choices[0].message:
            return chat_completion.choices[0].message.content.strip()  # Corrected access here
        else:
            return "No summary available."
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        return "Failed to retrieve summary due to an API error."