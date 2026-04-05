import streamlit as st
import re, random, pandas as pd, numpy as np, csv, warnings
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
# 🟢 NAYA NAAM YAHAN (Browser Tab ke liye)
st.set_page_config(page_title="AI Symptom Checker", page_icon="⚕️", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size:20px !important; color: #4CAF50;}
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-low { color: #00cc66; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHING DATA & MODEL
# ==========================================
@st.cache_resource
def load_and_train_model():
    try:
        training = pd.read_csv('Data/Deep_Cleaned_Training.csv') 
    except FileNotFoundError:
        try:
            training = pd.read_csv('Data/Super_Balanced_Training.csv')
        except FileNotFoundError:
            training = pd.read_csv('Data/Training.csv')
            
    testing = pd.read_csv('Data/Testing.csv')

    training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
    testing.columns  = testing.columns.str.replace(r"\.\d+$", "", regex=True)
    training = training.loc[:, ~training.columns.duplicated()]
    testing  = testing.loc[:, ~testing.columns.duplicated()]
    cols = training.columns[:-1]
    
    x = training[cols]
    y = training['prognosis']
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(x_train, y_train)

    symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}
    return model, le, cols, symptoms_dict, training

model, le, cols, symptoms_dict, training_data = load_and_train_model()

@st.cache_data
def load_dictionaries():
    description_list, severityDictionary, precautionDictionary = {}, {}, {}
    with open('MasterData/symptom_Description.csv') as csv_file:
        for row in csv.reader(csv_file):
            description_list[row[0]] = row[1]
            
    with open('MasterData/Symptom_severity.csv') as csv_file:
        for row in csv.reader(csv_file):
            try: severityDictionary[row[0]] = int(row[1])
            except: pass
            
    with open('MasterData/symptom_precaution.csv') as csv_file:
        for row in csv.reader(csv_file):
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
            
    return description_list, severityDictionary, precautionDictionary

description_list, severityDictionary, precautionDictionary = load_dictionaries()

# 🍏 NUTRITION & DIET DICTIONARY
nutrition_dict = {
    "headache": "💆 **Stress & Headache Relief Diet:** Stay hydrated with plenty of water. Drink herbal tea (peppermint or chamomile). Avoid excessive caffeine, screen time, and skip heavy/spicy meals.",
    "stomach": "🍌 **BRAT Diet Recommended:** Bananas, Rice, Applesauce, and Toast. Avoid dairy, spicy, and high-fat foods. Drink ORS.",
    "fever": "🍊 **Immunity Boost Diet:** High Vitamin C fruits (oranges, lemon), light soup, and ginger tea. Drink at least 3 liters of warm water.",
    "muscle": "🥚 **Protein & Recovery Diet:** Eggs, lentils, and spinach. Consume magnesium-rich foods like almonds to prevent cramps.",
    "diabetes": "🥗 **Low-GI Diet:** Complex carbs like oats. Avoid refined sugars. Include fenugreek (methi) and bitter gourd (karela).",
    "hypertension": "🧂 **DASH Diet:** Low sodium intake. Eat potassium-rich foods like spinach and bananas.",
    "default": "🍲 **Balanced Healing Diet:** Maintain a balanced diet rich in green vegetables, fresh fruits, and lean proteins. Stay hydrated."
}

# ==========================================
# 3. HELPER FUNCTIONS 
# ==========================================
symptom_synonyms = {
    "stomach ache":"stomach_pain", "belly pain":"stomach_pain", "tummy pain":"stomach_pain",
    "loose motion":"diarrhea", "motions":"diarrhea",
    "high temperature":"high_fever", "temperature":"high_fever", 
    "fever":"high_fever", "feaver":"high_fever", 
    "coughing":"cough", "throat pain":"sore_throat",
    "cold":"chills", "breathing issue":"breathlessness", "shortness of breath":"breathlessness",
    "body ache":"muscle_pain"
}

def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text: extracted.append(mapped)
    for symptom in all_symptoms:
        if symptom.replace("_"," ") in text: extracted.append(symptom)
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(word, [s.replace("_"," ") for s in all_symptoms], n=1, cutoff=0.8)
        if close:
            for sym in all_symptoms:
                if sym.replace("_"," ") == close[0]:
                    extracted.append(sym)
    return list(set(extracted))

def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class]*100,2)
    return disease, confidence

# ==========================================
# 4. CHAT LOGIC & STATE MACHINE
# ==========================================
def get_nutrition_advice(disease_category):
    return nutrition_dict.get(disease_category, nutrition_dict['default'])

def final_prediction():
    disease, conf = predict_disease(st.session_state.symptoms)
    user_symps_list = ' '.join(st.session_state.symptoms).lower()
    
    text = "### 🩺 Comprehensive Health & Nutrition Report\n---\n"
    
    if conf < 35.0 or len(st.session_state.symptoms) == 1:
        user_symps_str = ', '.join([s.replace('_', ' ') for s in st.session_state.symptoms])
        text += f"**Symptoms Analyzed:** {user_symps_str}\n\n🔎 **AI Confidence:** {conf}%\n\n"
        
        if 'headache' in user_symps_list or 'head' in user_symps_list or 'migraine' in user_symps_list:
            text += "🛑 **Diagnosis:** Tension Headache, Stress, or Mild Dehydration.\n"
            precautions = ["Rest in a quiet, dark room.", "Drink plenty of water.", "Massage your forehead and neck gently.", "Reduce screen time and sleep properly."]
            diet_cat = "headache"
            
        elif any(word in user_symps_list for word in ['stomach', 'vomiting', 'nausea', 'diarrhoea', 'belly', 'acidity', 'loose motion']):
            text += "🛑 **Diagnosis:** Mild stomach upset or gastrointestinal discomfort.\n"
            precautions = ["Drink plenty of fluids (ORS, coconut water).", "Eat light, easily digestible food.", "Avoid spicy or outside food.", "Consult a doctor if severe cramps occur."]
            diet_cat = "stomach"
            
        elif any(word in user_symps_list for word in ['muscle', 'joint', 'body ache', 'back pain', 'cramp', 'stiff', 'weakness', 'pain']):
            text += "🛑 **Diagnosis:** Minor strain, muscle fatigue, or weakness.\n"
            precautions = ["Take proper rest and avoid strenuous activity.", "Use a hot or cold compress.", "Maintain a good posture.", "Consult a physiotherapist if pain persists."]
            diet_cat = "muscle"
            
        else:
            text += "🛑 **Diagnosis:** Common Viral Infection or General Fatigue.\n"
            precautions = ["Rest and stay hydrated.", "Monitor your temperature.", "Eat light, healthy food.", "Consult a doctor if symptoms persist."]
            diet_cat = "fever"
            
    else:
        about = description_list.get(disease, 'No description available.')
        precautions = precautionDictionary.get(disease, [])
        text += f"🛑 **Diagnosis:** You may have **{disease}**\n\n🔎 **AI Confidence:** {conf}%\n\n📖 **About:** {about}\n"
        
        diet_cat = "default"
        if 'diabetes' in disease.lower(): diet_cat = "diabetes"
        elif 'hypertension' in disease.lower(): diet_cat = "hypertension"
        elif 'headache' in user_symps_list: diet_cat = "headache"
        elif any(word in user_symps_list for word in ['fever', 'cold', 'cough']): diet_cat = "fever"
        elif any(word in user_symps_list for word in ['stomach', 'diarrhoea']): diet_cat = "stomach"

    if precautions:
        text += "\n#### 🛡️ Suggested Precautions:\n"
        for i, p in enumerate(precautions):
            if p.strip(): text += f"- {p.capitalize()}\n"

    nutrition = get_nutrition_advice(diet_cat)
    text += f"\n#### 🥗 AI Nutrition & Diet Plan:\n{nutrition}\n"

    text += f"\n---\n💡 *A healthy outside starts from the inside. Take care, {st.session_state.user_data.get('name', 'Friend')}!*"
    return text

def ask_next_symptom():
    i = st.session_state.ask_index
    ds = st.session_state.disease_syms
    if i < min(8, len(ds)):
        sym = ds[i]
        st.session_state.ask_index += 1
        return f"👉 Do you also experience **{sym.replace('_',' ')}**? (Yes/No):"
    else:
        st.session_state.step = 'final'
        return final_prediction()

# ==========================================
# 5. UI INITIALIZATION & SIDEBAR
# ==========================================
if "messages" not in st.session_state:
    # 🟢 NAYA NAAM YAHAN (Welcome Message)
    st.session_state.messages = [{"role": "assistant", "content": "🤖 Welcome to the AI Symptom Checker!\n\n👉 Let's start. What is your name?"}]
    st.session_state.step = 'name'
    st.session_state.symptoms = []
    st.session_state.disease_syms = []
    st.session_state.ask_index = 0
    st.session_state.initial_conf = 0.0
    st.session_state.pred_disease = ""
    st.session_state.user_data = {}
    st.session_state.progress = 10

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.header("👤 Patient Profile")
    st.divider()
    u_data = st.session_state.user_data
    st.markdown(f"**Name:** {u_data.get('name', 'N/A')}")
    st.markdown(f"**Age:** {u_data.get('age', 'N/A')} | **Gender:** {u_data.get('gender', 'N/A')}")
    st.markdown(f"**Severity (1-10):** {u_data.get('severity', 'N/A')}")
    
    if 'severity' in u_data:
        try:
            sev = int(u_data['severity'])
            if sev >= 7: st.markdown("<p class='risk-high'>⚠️ High Risk Indicator</p>", unsafe_allow_html=True)
            elif sev >= 4: st.markdown("<p style='color:orange;'>⚠️ Moderate Risk</p>", unsafe_allow_html=True)
            else: st.markdown("<p class='risk-low'>✅ Low Risk</p>", unsafe_allow_html=True)
        except: pass
    st.divider()
    # 🟢 NAYA NAAM YAHAN (Sidebar Caption)
    st.caption("AI Symptom Checker v2.0\nDeveloped by Akash & Aashu")

# 🟢 NAYA NAAM YAHAN (Main Title)
st.title("🤖 AI Symptom Checker")
st.progress(st.session_state.progress / 100.0, text="Diagnosis Progress")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your answer here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    step = st.session_state.step
    reply = ""

    if step == 'name':
        st.session_state.user_data['name'] = prompt
        reply = f"Hi {prompt}! 👉 Please enter your age:"
        st.session_state.step = 'age'
        st.session_state.progress = 20
        
    elif step == 'age':
        st.session_state.user_data['age'] = prompt
        reply = "👉 What is your gender? (M/F/Other):"
        st.session_state.step = 'gender'
        st.session_state.progress = 30
        
    elif step == 'gender':
        st.session_state.user_data['gender'] = prompt
        reply = "👉 Please describe your symptoms (even 1 symptom is fine):"
        st.session_state.step = 'symptoms'
        st.session_state.progress = 40
        
    elif step == 'symptoms':
        symptoms_list = extract_symptoms(prompt, cols)
        if not symptoms_list:
            reply = "❌ Could not detect valid symptoms. Please describe again using common terms:"
        else:
            st.session_state.symptoms = symptoms_list
            disease, conf = predict_disease(symptoms_list)
            st.session_state.pred_disease = disease
            st.session_state.initial_conf = conf
            reply = f"✅ Symptoms Noted: **{', '.join(symptoms_list).replace('_', ' ')}**\n\n👉 For how many days have you had these symptoms?"
            st.session_state.step = 'days'
            st.session_state.progress = 60
            
    elif step == 'days':
        st.session_state.user_data['days'] = prompt
        reply = "👉 On a scale of 1–10 (10 being worst), how severe is your condition?"
        st.session_state.step = 'severity'
        st.session_state.progress = 70
        
    elif step == 'severity':
        st.session_state.user_data['severity'] = prompt
        reply = "👉 Do you have any pre-existing medical conditions? (e.g., Asthma, BP, No):"
        st.session_state.step = 'preexist'
        st.session_state.progress = 80
        
    elif step == 'preexist':
        st.session_state.user_data['preexist'] = prompt
        reply = "👉 Do you smoke, drink alcohol, or have irregular sleep? (Yes/No):"
        st.session_state.step = 'lifestyle'
        st.session_state.progress = 90
        
    elif step == 'lifestyle':
        st.session_state.user_data['lifestyle'] = prompt
        reply = "👉 Any family history of similar illness? (Yes/No):"
        st.session_state.step = 'family'
        
    elif step == 'family':
        st.session_state.user_data['family'] = prompt
        
        if st.session_state.initial_conf < 30.0:
            reply = final_prediction()
            st.session_state.step = 'final'
            st.session_state.progress = 100
        else:
            disease = st.session_state.pred_disease
            raw_disease_syms = list(training_data[training_data['prognosis'] == disease].iloc[0][:-1].index[
                training_data[training_data['prognosis'] == disease].iloc[0][:-1] == 1
            ])
            
            filtered_symps = [sym for sym in raw_disease_syms if sym not in st.session_state.symptoms]
            
            st.session_state.disease_syms = filtered_symps
            st.session_state.ask_index = 0
            st.session_state.step = 'guided'
            reply = ask_next_symptom()
            
    elif step == 'guided':
        idx = st.session_state.ask_index - 1
        if 0 <= idx < len(st.session_state.disease_syms):
            if prompt.strip().lower() in ['yes', 'y', 'haan']:
                st.session_state.symptoms.append(st.session_state.disease_syms[idx])
        reply = ask_next_symptom()
        if st.session_state.step == 'final': st.session_state.progress = 100
        
    elif step == 'final':
        reply = "🩺 The consultation is complete! Please refresh the page (or press F5) to start a new diagnosis."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()