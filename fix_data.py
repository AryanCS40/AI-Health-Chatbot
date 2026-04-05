import pandas as pd
import warnings

warnings.filterwarnings("ignore")

print("🧹 Dataset ki Deep Cleaning shuru ho rahi hai...")

try:
    # 1. Purana dataset load karo (apna path check kar lena)
    df = pd.read_csv('Data/Training.csv')
    
    # 2. Columns ke naam ke aage-peechhe ke spaces hatao aur beech ke spaces ko '_' banao
    df.columns = df.columns.str.strip() 
    df.columns = df.columns.str.replace(r'\s+', '_', regex=True) 
    
    # 3. Duplicate rows hatao taaki AI over-smart na bane
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"🗑️ {initial_rows - len(df)} duplicate rows hata di gayi hain!")

    # 4. Bimaari ke naam (prognosis) mein se faltu spaces hatao
    df['prognosis'] = df['prognosis'].str.strip()

    # 5. Common bimaariyan add karo taaki bot seedha 'AIDS' ya 'Paralysis' na bole
    cols = df.columns[:-1] 
    
    def create_safe_disease(disease_name, symptom_list, count=100):
        rows = []
        for _ in range(count):
            row = {col: 0 for col in cols}
            for sym in symptom_list:
                clean_sym = sym.strip().replace(' ', '_')
                if clean_sym in row:
                    row[clean_sym] = 1
            row['prognosis'] = disease_name
            rows.append(row)
        return rows

    print("💉 Common bimaariyan (Viral Fever, Headache) dataset mein daal rahe hain...")
    viral_fever = create_safe_disease('Viral Fever', ['high_fever', 'headache', 'fatigue', 'chills', 'muscle_pain', 'malaise'])
    tension_headache = create_safe_disease('Tension Headache', ['headache', 'fatigue', 'dizziness', 'neck_pain'])
    stomach_upset = create_safe_disease('Mild Stomach Upset', ['vomiting', 'stomach_pain', 'nausea', 'diarrhoea', 'acidity'])

    new_data = pd.DataFrame(viral_fever + tension_headache + stomach_upset)
    df = pd.concat([df, new_data], ignore_index=True)

    # 6. Extreme bimaariyon par 'Lock' lagao (Strict Rules)
    if 'extra_marital_contacts' in df.columns:
        df.loc[df['prognosis'] == 'AIDS', 'extra_marital_contacts'] = 1
    if 'weakness_of_one_body_side' in df.columns:
        df.loc[df['prognosis'] == 'Paralysis (brain hemorrhage)', 'weakness_of_one_body_side'] = 1

    # 7. Naya, ekdum saaf Dataset save karo
    df.to_csv('Data/Deep_Cleaned_Training.csv', index=False)
    
    print("✅ BUMPER SUCCESS! 'Deep_Cleaned_Training.csv' ban gaya hai.")
    print("👉 Ab apne main Chatbot wale code mein is nayi file ko load karna!")

except FileNotFoundError:
    print("❌ Error: 'Data/Training.csv' nahi mila. Folder ka path check kar lo bhai.")
except Exception as e:
    print(f"❌ Oops! Ek error aa gaya: {e}")