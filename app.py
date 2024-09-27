import nltk
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import logging

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load sentence transformer for semantic similarity
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

app = Flask(__name__)

# Predefined FAQs with keyword mapping
faqs = {
    "What is IAC?": {
        "response": "The IAC Internship Program connects students and freshers with industry opportunities to help them become job-ready.",
        "keywords": ["iac", "internship", "program"]
    },
    "Hello": {
        "response": "Hello! How can I assist you today?",
        
    },
    "hello": {
        "response": "Hello! How can I assist you today?",
        
    },
    "Hi": {
        "response": "Hello! How can I assist you today?",
        
    },
    "hi": {
        "response": "Hello! How can I assist you today?",
        
    },
    "Who are you?": {
        "response": "I am an AI chatbot here to assist you with the IAC Internship Program and other queries you may have.",
        "keywords": ["who", "are", "you", "chatbot"]
    },
    "Thank you": {
        "response": "You're welcome! Feel free to ask if you have any more questions.",
        "keywords": ["thank", "thanks", "thank you"]
    },
    "Goodbye": {
        "response": "Goodbye! Have a great day!",
        "keywords": ["goodbye", "bye", "see you", "farewell"]
    },
    "What is IAC?": {
        "response": "The IAC Internship Program connects students and freshers with industry opportunities to help them become job-ready.",
        "keywords": ["iac", "internship", "program"]
    },
    "What is the Industry Academia Community (IAC) Internship Program?": {
        "response": "The IAC Internship Program connects students and freshers with industry opportunities to help them become job-ready.",
        "keywords": ["what", "industry", "academia", "community", "iac", "internship", "program"]
    },
    "Who can participate in the IAC Internship Program?": {
        "response": "The program is open to students and recent graduates looking for internship experiences in various fields.",
        "keywords": ["who", "participate", "iac", "internship", "program", "join"]
    },
    "What can the chatbot help me with?": {
        "response": "The chatbot can answer queries related to the internship program, including application processes, eligibility criteria, and internship details.",
        "keywords": ["what", "chatbot", "help", "internship", "program"]
    },
    "Is the chatbot available on multiple platforms?": {
        "response": "Yes, the chatbot is integrated with Facebook, Instagram, LinkedIn, WhatsApp, and SMS for easy access.",
        "keywords": ["available", "multiple", "platforms"]
    },
    "Can I interact with the chatbot using voice?": {
        "response": "Yes, the chatbot supports both text and voice interactions for a more user-friendly experience.",
        "keywords": ["interact", "voice"]
    },
    "What technology is used to build the chatbot?": {
        "response": "The chatbot is developed using Python and leverages popular machine learning and data science libraries for enhanced functionality.",
        "keywords": ["technology", "build", "language", "used"]
    },
    "How are FAQs managed in the chatbot?": {
        "response": "FAQs are pre-loaded into the chatbot's backend, allowing for quick and accurate responses to common queries.",
        "keywords": ["faqs", "managed"]
    },
    "How quickly can I expect a response from the chatbot?": {
        "response": "The chatbot is designed to provide instant responses, significantly reducing waiting times compared to traditional customer support.",
        "keywords": ["quickly", "expect", "response"]
    },
    "What should I do if the chatbot cannot answer my question?": {
        "response": "If the chatbot cannot address your query, you will be directed to a human representative for further assistance.",
        "keywords": ["do", "cannot", "answer", "question"]
    },
    "Can I provide feedback on the chatbot's performance?": {
        "response": "Yes, user feedback is encouraged to continuously improve the chatbot's capabilities and responsiveness.",
        "keywords": ["provide", "feedback", "performance"]
    },
    "How often is the information in the chatbot updated?": {
        "response": "The chatbot's FAQs and information are regularly reviewed and updated to ensure accuracy and relevance.",
        "keywords": ["how", "often", "information", "updated"]
    },
    "Who do I contact for further inquiries?": {
        "response": "For additional questions or concerns, please reach out to the IAC Success Team through the provided contact methods on our website.",
        "keywords": ["who", "contact", "further", "inquiries"]
    },
    "How do I apply for an internship through the IAC program?": {
        "response": "You can apply through our website or via the chatbot, which will guide you through the application process.",
        "keywords": ["apply", "internship", "iac", "program"]
    },
    "What documents do I need to submit with my application?": {
        "response": "Typically, you will need a resume, cover letter, and any relevant academic transcripts or certificates.",
        "keywords": ["documents", "need", "submit", "application"]
    },
    "When will I hear back about my application status?": {
        "response": "The response time may vary, but you can expect to hear back within 1-2 weeks after submitting your application.",
        "keywords": ["hear", "back", "application", "status"]
    },
    "What types of internships are available?": {
        "response": "Internships span various fields, including technology, marketing, finance, and more. The chatbot can provide specific details based on your interests.",
        "keywords": ["types", "internships", "available"]
    },
    "Are internships paid or unpaid?": {
        "response": "Internship compensation varies by company and position; the chatbot can help you find this information for specific roles.",
        "keywords": ["internships", "paid", "unpaid"]
    },
    "What skills are required for the internships?": {
        "response": "Required skills vary by internship, but common skills include teamwork, communication, and technical proficiency. The chatbot can suggest roles based on your skills.",
        "keywords": ["skills", "required", "internships"]
    },
    "Is my information secure when using the chatbot?": {
        "response": "Yes, we take user privacy seriously. All interactions are encrypted and your personal information is protected.",
        "keywords": ["information", "secure", "using", "chatbot"]
    },
    "Can I ask the chatbot about specific companies offering internships?": {
        "response": "Yes, the chatbot can provide information about companies participating in the IAC program and their internship offerings.",
        "keywords": ["ask", "specific", "companies", "internships"]
    },
    "What should I do if the chatbot is not responding?": {
        "response": "If you encounter issues, try refreshing the page or restarting the chat. If problems persist, please contact the IAC Success Team for assistance.",
        "keywords": ["not", "responding", "chatbot"]
    },
    "Can I access the chatbot from mobile devices?": {
        "response": "Yes, the chatbot is optimized for both desktop and mobile use, allowing you to access it from any device.",
        "keywords": ["access", "mobile", "devices"]
    },
    "Does the IAC program provide any training or resources for interns?": {
        "response": "Yes, participants can access various resources and training materials to enhance their skills. The chatbot can guide you to these resources.",
        "keywords": ["iac", "program", "training", "resources"]
    },
    "Can I ask the chatbot for career advice?": {
        "response": "Absolutely! The chatbot can provide general career tips and advice tailored to your field of interest.",
        "keywords": ["ask", "career", "advice"]
    },
    "Will I have the opportunity to network with other interns?": {
        "response": "Yes, the program often includes networking events and opportunities to connect with fellow interns and industry professionals.",
        "keywords": ["network", "interns"]
    },
    "How can I stay updated on future internship opportunities?": {
        "response": "You can subscribe to our newsletter or follow our social media channels for the latest updates. The chatbot can help you with the subscription process.",
        "keywords": ["stay", "updated", "future", "internship", "opportunities"]
    },
    "Will the chatbot evolve with user feedback?": {
        "response": "Yes, we are committed to continuous improvement based on user feedback, and updates will be implemented regularly.",
        "keywords": ["evolve", "user", "feedback"]
    },
    "How can I suggest new features for the chatbot?": {
        "response": "You can provide suggestions through the feedback option in the chatbot interface, and we’ll consider them for future updates.",
        "keywords": ["suggest", "features", "chatbot"]
    },
    "How can I prepare for an internship interview?": {
        "response": "The chatbot can provide tips on common interview questions, attire recommendations, and best practices for presenting yourself professionally.",
        "keywords": ["prepare", "internship", "interview"]
    },
    "Are there any workshops available to help with resume writing?": {
        "response": "Yes, the IAC program offers workshops on resume writing and interview skills. The chatbot can provide information on upcoming sessions.",
        "keywords": ["workshops", "resume", "writing"]
    },
    "Can the chatbot provide updates on my application status?": {
        "response": "Yes, the chatbot can provide general updates. For specific application details, please contact the IAC Success Team directly.",
        "keywords": ["updates", "application", "status"]
    },
    "Is there a way to save my chat history with the chatbot?": {
        "response": "Currently, the chatbot does not support saving chat history. However, you can take screenshots of important information.",
        "keywords": ["save", "chat", "history", "chatbot"]
    },
    "What happens if I miss a deadline for an application?": {
        "response": "If you miss a deadline, you may need to wait for the next application cycle. The chatbot can guide you through upcoming deadlines.",
        "keywords": ["miss", "deadline", "application"]
    },
    "Can I apply for multiple internships at the same time?": {
        "response": "Yes, you can apply for multiple internships simultaneously, but ensure you meet the requirements for each position.",
        "keywords": ["apply", "multiple", "internships", "same", "time"]
    },
    "Is there a limit to how many internships I can apply for?": {
        "response": "Generally, there is no limit to the number of internships you can apply for, but it's best to prioritize quality over quantity.",
        "keywords": ["limit", "internships", "apply"]
    },
    "What criteria are used to select interns for the program?": {
        "response": "Selection criteria may include academic performance, skills, relevant experience, and interview performance.",
        "keywords": ["criteria", "select", "interns", "program"]
    },
    "How can I find out more about companies participating in the IAC program?": {
        "response": "The chatbot can provide a list of participating companies and details about their internship offerings.",
        "keywords": ["find", "out", "more", "companies", "participating", "iac", "program"]
    }
}

# Keyword synonym mapping
keyword_synonym_mapping = {
    "join": ["participate", "enroll", "take part"],
    "apply": ["submit", "register", "enroll", "sign up"],
    "help": ["assist", "support", "aid"],
    "who": ["which", "anyone", "whom"],
    "what": ["which", "details", "information"],
    "can": ["is able to", "able to", "could"],
    "do": ["perform", "execute", "carry out"]
}

# Convert FAQ questions into embeddings
faq_questions = list(faqs.keys())
faq_embeddings = sentence_model.encode(faq_questions, convert_to_tensor=True)

# Function to preprocess input text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

# Function to get synonyms for words using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# Function to expand user query with synonyms
def expand_query_with_synonyms(query):
    tokens = word_tokenize(query)
    expanded_tokens = []
    for token in tokens:
        if token in keyword_synonym_mapping:
            expanded_tokens.extend(keyword_synonym_mapping[token])
        else:
            expanded_tokens.append(token)
            synonyms = get_synonyms(token)
            expanded_tokens.extend(list(synonyms))  # Add WordNet synonyms if they exist
    return " ".join(expanded_tokens)

# Store the last few user queries
user_query_history = []

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
@app.route("/get", methods=["POST"])
@app.route("/get", methods=["POST"])
@app.route("/get", methods=["POST"])
def chat():
    global user_query_history
    msg = request.form["msg"]
    logging.debug(f"User Message: {msg}")

    # Preprocess the message with synonym expansion
    msg = preprocess_text_with_synonyms(msg)
    user_query_history.append(msg)

    # Check for greetings, thanks, goodbyes, identity questions, and basic questions using keywords
    if any(greet in msg.lower() for greet in ["hello", "hi", "hey"]):
        response = "Hello! How can I assist you today?"
    elif any(thanks in msg.lower() for thanks in ["thank you", "thanks"]):
        response = "You're welcome! Feel free to ask if you have any more questions."
    elif any(goodbye in msg.lower() for goodbye in ["goodbye", "bye", "see you", "farewell"]):
        response = "Have a great day!"
    elif any(keyword in msg.lower() for keyword in ["who", "are", "you", "what", "name", "tell"]):
        response = "I'm a chatbot designed to assist you with your queries only realted to IAC. How can I help you today?"
    elif any(keyword in msg.lower() for keyword in [
        "purpose", "what", "do", "help", "function", "capability", "assist"
    ]):
        response = "My purpose is to assist you by providing information about IAC and answering your questions to the best of my ability."
    elif any(keyword in msg.lower() for keyword in [
        "name", "who", "made", "age", "where", "from"
    ]):
        response = "I'm just a virtual assistant created to help you with your queries. I don't have a physical form or age."
    else:
        response = get_contextual_response(msg, user_query_history)
        logging.debug(f"Response: {response}")

        if response is None:
            response = "I'm not sure about that. Could you clarify your question or ask about something else?"

    return jsonify({"response": response})



# Preprocess text and expand it with synonyms
def preprocess_text_with_synonyms(text):
    text = preprocess_text(text)  # Existing preprocessing
    return expand_query_with_synonyms(text)

# Function to get a contextual response based on current query and history
def get_contextual_response(current_query, query_history):
    global faq_embeddings
    # Use only the last two queries to maintain context
    context = " ".join(query_history[-2:]) + " " + current_query
    user_embedding = sentence_model.encode(context, convert_to_tensor=True)

    # Compute similarities with the FAQ embeddings
    similarities = util.pytorch_cos_sim(user_embedding, faq_embeddings)[0]

    # Get the most similar FAQ based on similarity score
    max_similarity_score, max_index = torch.max(similarities, dim=0)
    threshold = 0.6  # Adjust this threshold as needed for precision

    if max_similarity_score >= threshold:
        return faqs[faq_questions[max_index]]["response"]

    # If no direct match is found
    return None

if __name__ == "__main__":
    app.run(debug=True,port=5001)
