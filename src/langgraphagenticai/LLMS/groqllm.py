import os
import streamlit as st
from langchain_groq import ChatGroq

class GroqLLM:
    def __init__(self,user_controls_input):
        self.user_controls_input=user_controls_input

    def get_llm_SDLC_model_model(self):
        try:
            groq_api_key=self.user_controls_input['GROQ_API_KEY']
            selected_groq_model=self.user_controls_input['selected_groq_model']
            if groq_api_key=='' and os.environ["GROQ_API_KEY"] =='':
                st.error("Please Enter the Groq API KEY")

            llm = ChatGroq(api_key =groq_api_key, model=selected_groq_model)

        except Exception as e:
            raise ValueError(f"Error Occurred with Exception : {e}")
        return llm
    
    def get_llm_SDLC_model(temperature=0.7, streaming=False, streaming_callback=None):
        """
        Get a Groq LLM instance.

        Args:
            temperature (float): The temperature for generation.
            streaming (bool): Whether to stream the response.
            streaming_callback (Callable[[str], None]): Callback function for streaming.

        Returns:
            ChatGroq: The LLM instance.
        """
        # Get API key from environment
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not found. Please make sure it's set.")

        # Initialized LLM with model
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-8b-8192",  # Use Llama 3 8B model for high quality results
            temperature=temperature,
            streaming=streaming,
        )

        return llm

       