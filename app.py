import streamlit as st
import os
import json
from typing import Literal
from pydantic import BaseModel
from litellm import completion

# Configure the visual style of the web page
st.set_page_config(page_title="AI Emotion Council", page_icon="⚖️", layout="wide")

st.title("⚖️ The 3-LLM Emotion Council")
st.markdown("Enter a sentence. Three different free AI models will evaluate it and vote on the underlying emotion.")

# We define a strict schema. This forces all 3 models to output the exact same JSON keys.
class EmotionAnalysis(BaseModel):
    emotion: Literal["joy", "sadness", "angry", "neutral", "surprise", "fear"]
    confidence_score: float  # Scale of 0.0 to 1.0
    reasoning: str          # A brief sentence explaining why

def get_model_opinion(model_path: str, api_key: str, sentence: str) -> dict:
    """Helper function to run unified API requests across different free models."""
    prompt = f"Analyze the emotion of this sentence: \"{sentence}\""
    try:
        response = completion(
            model=model_path,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            response_format=EmotionAnalysis, # Enforce our Pydantic schema structure
            temperature=0.0                  # Standardizes the logic
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

# User Interface input
user_sentence = st.text_input("Enter a sentence for the council to analyze:", "I can't believe you did this for me!")

if st.button("Consult Council", type="primary"):
    # Grab the API keys securely from the cloud settings
    gemini_key = os.getenv("GEMINI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if not gemini_key or not openrouter_key:
        st.error("API Keys are missing! Please configure them in your Host Settings.")
    else:
        # Define our three completely free models
        # Define our updated, active free models
        # Define our three completely active, free models (All via OpenRouter!)
        # Updated stable free models on OpenRouter
        # The new stable, 100% free council (Including Gemini!)
        # The ultimate stable free council (Uses the OpenRouter Auto-Free Router for Slot 3!)
        # Grab the Groq API key alongside OpenRouter
        groq_key = os.getenv("GROQ_API_KEY")

        # The Ultimate Multi-Provider Stable Free Council
        models = {
            "OpenAI GPT-OSS 20B (Free)": {
                "path": "openrouter/openai/gpt-oss-20b:free",
                "key": openrouter_key
            },
            "Tencent Hy3 MoE (Free)": {
                "path": "openrouter/tencent/hy3:free",
                "key": openrouter_key
            },
            "Meta Llama 3.3 70B (Groq Free)": {
                "path": "groq/llama-3.3-70b-specdec",
                "key": groq_key
            }
        }
        
        # Display a loading spinner
        with st.spinner("Analyzing sentence... please wait..."):
            opinions = {}
            for name, config in models.items():
                opinions[name] = get_model_opinion(config["path"], config["key"], user_sentence)
        
        # Create 3 clean columns to display results side-by-side
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        votes = []
        
        for i, (name, result) in enumerate(opinions.items()):
            with cols[i]:
                st.subheader(name)
                if "error" in result:
                    st.error(f"Error calling model: {result['error']}")
                else:
                    votes.append(result['emotion'])
                    # Render a clean UI card for each model
                    st.metric(label="Predicted Emotion", value=result['emotion'].upper())
                    st.progress(result['confidence_score'])
                    st.write(f"**Confidence:** {int(result['confidence_score'] * 100)}%")
                    st.info(f"💬 *\"{result['reasoning']}\"*")
                    
        # Consensus Calculations
        st.markdown("---")
        if votes:
            most_common = max(set(votes), key=votes.count)
            agreement_count = votes.count(most_common)
            
            if agreement_count == 3:
                st.balloons()
                st.success(f"🎉 **Unanimous Agreement:** All models agreed the emotion is **{most_common.upper()}**!")
            elif agreement_count == 2:
                st.info(f"🤝 **Majority Consensus:** 2 out of 3 models agreed the emotion is **{most_common.upper()}**.")
            else:
                st.warning("⚖️ **No Consensus:** The council split! Every model picked a different emotion.")
