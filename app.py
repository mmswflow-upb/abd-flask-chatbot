import os
from flask import Flask, request, jsonify
from textblob import TextBlob
import google.generativeai as genai

########################################
# Configuration
########################################
genai.configure(api_key="AIzaSyAso1BzoBSxR4C2bPN8IOHbs3lYMvQ1euc")
model = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM_INSTRUCTION_BASE = (
    "You are a highly supportive and empathetic mental health chatbot with advanced reasoning capabilities. "
    "You have memory of the conversation and user details through provided summaries. You are not a "
    "professional therapist, but you can offer empathy, understanding, and direct users to reputable resources. "
    "You encourage seeking professional help when appropriate, never give medical diagnoses, and handle crisis "
    "situations by providing immediate hotline resources. If the user is off-topic, gently steer them back to "
    "discussing their feelings. If the user requests a final conclusion, provide a comprehensive summary of the "
    "situation and helpful next steps."
)

DISCLAIMER = (
    "I am not a licensed professional. If you’re feeling overwhelmed or in crisis, please reach out to "
    "a mental health professional or call your local emergency number."
)

RESOURCE_LINKS = {
    "general_support": "https://www.mentalhealth.gov/",
    "therapy_locator": "https://www.apa.org/helpcenter/find-therapist",
    "crisis_hotline_usa": "https://988lifeline.org/ (USA)",
    "crisis_hotline_international": "https://www.iasp.info/resources/Crisis_Centres/"
}

CRISIS_KEYWORDS = ["suicide", "kill myself", "end my life", "not worth living", "overdose"]
DEPRESSED_THRESHOLD = -0.5

########################################
# Helper Functions
########################################

def analyze_message_sentiment(message: str) -> float:
    analysis = TextBlob(message)
    return analysis.sentiment.polarity

def check_for_crisis_indicators(message: str) -> bool:
    lower_msg = message.lower()
    return any(keyword in lower_msg for keyword in CRISIS_KEYWORDS)

def is_depressed(user_input: str) -> bool:
    sentiment = analyze_message_sentiment(user_input)
    return sentiment < DEPRESSED_THRESHOLD

def classify_message(user_input: str) -> str:
    """
    Classify the user's message into categories such as 'on-topic' or 'off-topic' using the LLM.
    This classification helps the assistant decide whether to redirect the conversation.
    """
    classification_prompt = (
        "You are a classifier. The user message will be given, and you must respond with one label: 'on-topic' or 'off-topic'. "
        "Consider 'on-topic' if the user message relates to emotions, mental health, feelings, or their personal struggles. "
        "Consider 'off-topic' if the user message is unrelated to emotional well-being.\n\n"
        f"User Message: {user_input}\nRespond with only 'on-topic' or 'off-topic'."
    )
    response = model.generate_content(classification_prompt)
    classification = response.text.strip().lower()
    return classification if classification in ["on-topic", "off-topic"] else "on-topic"

def update_summary(summary: str, conversation_history: list) -> str:
    """
    Update the conversation summary. This summary focuses on the emotional trajectory,
    main concerns, and user progress. The summary is concise but reflective of key details.
    """
    recent_turns = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    summary_prompt = (
        "Update the following summary of the conversation so far, focusing on the user's emotional state, "
        "primary concerns, goals, and any progress or guidance provided.\n\n"
        f"Current Summary: {summary}\n\n"
        "Recent Turns:\n"
    )
    for turn in recent_turns:
        summary_prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    summary_prompt += "\nUpdate the summary above based on these recent turns, keeping it concise, clear, and focused on emotional aspects."

    response = model.generate_content(summary_prompt)
    return response.text.strip()

def update_user_biography(user_bio: str, conversation_history: list) -> str:
    """
    Update a user biography memory that extracts personal details the user has mentioned: 
    their general situation, any personal stressors, hints about their life context, coping strategies they've tried, etc.
    This helps the assistant remember who the user is and what they've shared.
    """
    recent_turns = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    bio_prompt = (
        "You are maintaining a concise 'user biography' that stores factual details the user shares about themselves: "
        "This includes their situation, triggers, coping strategies, what helps or doesn’t help, any mention of friends, family, or resources they've tried. "
        "Do not include emotional reflections, just factual info. Update the biography below with any new factual details.\n\n"
        f"Current Biography: {user_bio}\n\n"
        "Recent Turns:\n"
    )
    for turn in recent_turns:
        bio_prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    bio_prompt += "\nUpdate the biography above based on these recent turns. Keep it factual and concise."

    response = model.generate_content(bio_prompt)
    return response.text.strip()

def build_prompt(summary, user_bio, user_message, system_instruction):
    prompt = (
        f"{system_instruction}\n\n"
        f"CONTEXT SUMMARY:\n{summary}\n\n"
        f"USER BIOGRAPHY:\n{user_bio}\n\n"
        f"USER MESSAGE: {user_message}\n"
        "ASSISTANT:"
    )
    return prompt

def user_requests_conclusion(user_input: str) -> bool:
    phrases = ["final result", "conclusion", "summary", "sum up", "end advice", "final advice"]
    return any(phrase in user_input.lower() for phrase in phrases)

########################################
# Flask Application
########################################

app = Flask(__name__)

# Global variables to maintain conversation state
conversation_history = []
summary = "The user has just started sharing. No significant details yet."
user_bio = "No personal details known yet."

@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history, summary, user_bio

    data = request.get_json()
    if not data or "user_message" not in data:
        return jsonify({"error": "Please provide 'user_message' in JSON."}), 400

    user_input = data["user_message"]

    # Check for crisis
    if check_for_crisis_indicators(user_input):
        assistant_reply = (
            "I’m so sorry you’re feeling this way. You’re not alone. If you’re "
            "thinking about suicide or harming yourself, please consider reaching out immediately. In the US, "
            "you can call or text 988 or visit 988lifeline.org for immediate support. Internationally, find "
            f"resources here: {RESOURCE_LINKS['crisis_hotline_international']}. Your life matters."
        )
        conversation_history.append({"user": user_input, "assistant": assistant_reply})
        return jsonify({"assistant": assistant_reply})

    depressed = is_depressed(user_input)
    classification = classify_message(user_input)
    wants_conclusion = user_requests_conclusion(user_input)

    # Adjust system instruction based on off-topic or final conclusion requests
    if classification == "off-topic" and not wants_conclusion:
        custom_instruction = (
            SYSTEM_INSTRUCTION_BASE + 
            " The user seems to be discussing something unrelated to their emotional well-being. "
            "Gently remind them of your purpose in supporting their mental health and encourage them to talk "
            "about their feelings or what’s troubling them."
        )
    else:
        custom_instruction = SYSTEM_INSTRUCTION_BASE

    # Build the prompt with current summaries and user input
    prompt = build_prompt(summary, user_bio, user_input, custom_instruction)
    response = model.generate_content(prompt)
    assistant_reply = response.text.strip()

    # Add resources if depressed
    if depressed:
        assistant_reply += (
            f"\n\nYou might find these resources helpful:\n- {RESOURCE_LINKS['general_support']}\n"
            f"- {RESOURCE_LINKS['therapy_locator']}\n"
            "Remember, you deserve compassion and understanding."
        )

    # If off-topic (and not concluding), gently steer back
    if classification == "off-topic" and not wants_conclusion:
        assistant_reply += (
            "\n\nIt seems we've drifted off from your feelings. I'm here to help and support you emotionally. "
            "What’s on your mind emotionally? Is there something troubling you or affecting how you feel?"
        )

    # Append to conversation history
    conversation_history.append({"user": user_input, "assistant": assistant_reply})

    # Update summaries
    summary = update_summary(summary, conversation_history)
    user_bio = update_user_biography(user_bio, conversation_history)

    return jsonify({
        "assistant": assistant_reply,
        "disclaimer": DISCLAIMER,
        "conversation_history": conversation_history,
        "summary": summary,
        "user_bio": user_bio
    })


if __name__ == "__main__":
    # Run the Flask app on http://127.0.0.1:5000/chat
    app.run(host="0.0.0.0", port=5000, debug=True)
