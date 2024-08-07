# 💬 Baby Nutrition Advisor Chatbot

A simple app that provides answers regarding baby / infant nutrition.

The app is built on Streamlit, uses LlamaIndex and OpenAi's ChatGPT 4o-mini model

The data is scraped from public goverment websites: 
- [NHS UK](https://www.nhs.uk/conditions/baby/weaning-and-feeding/)
- [US Department of Agriculture](https://www.myplate.gov/life-stages/infants)
- [US Centers for Disease Control](https://www.cdc.gov/nutrition/infantandtoddlernutrition/index.html)
- [Mayoclinic](https://www.mayoclinic.org/healthy-lifestyle/infant-and-toddler-health/in-depth/breastfeeding-nutrition/art-20046912)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://infantnutritionchatbot.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run main.py
   ```

3. Running the supporting functions; question generation and evaluation, requires the openAi_API key to be set up as an environment variable.
