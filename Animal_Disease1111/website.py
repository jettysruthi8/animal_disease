import streamlit as st
import joblib
import pickle
import numpy as np


# Load Models
disease_model = joblib.load("disease_prediction_model_new.pkl")
danger_model = joblib.load("danger_prediction_model_new.pkl")

# Load Label Encoders
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Disease Descriptions
disease_info = {
    "Canine Parvovirus": "A highly contagious viral disease in dogs that causes severe vomiting and diarrhea.",
    "Canine Coronavirus": "A viral infection affecting dogs, leading to mild gastroenteritis and respiratory issues.",
    "Canine Distemper": "A serious viral disease causing fever, nasal discharge, and neurological problems.",
    "Canine Influenza": "A respiratory infection in dogs with symptoms like coughing, fever, and nasal discharge.",
    "Infectious canine hepatitis": "A viral disease affecting the liver, kidneys, and eyes, leading to fever and vomiting.",
    "Pseudorabies": "A viral disease that affects the nervous system, causing itching and respiratory distress.",
    "Mast Cell Tumor": "A type of skin cancer in dogs that may appear as lumps or swellings on the skin.",
    "Melonoma(Mouth)": "An aggressive oral cancer in dogs, often found in the gums or mouth tissue.",
    "Lymphoma": "A common cancer in dogs affecting the lymph nodes, causing swelling and lethargy.",
    "Osteosarcoma": "A bone cancer in dogs, commonly affecting the legs and leading to limping and pain.",
    "Hemangiosarcoma": "A cancer of the blood vessels, often affecting the spleen or heart, causing internal bleeding.",
    "Brucellosis": "A bacterial infection in dogs that can cause reproductive issues and fever.",
    "Leptospirosis": "A bacterial infection spread through contaminated water, leading to kidney and liver damage.",
    "Lyme": "A tick-borne disease that causes fever, joint pain, and lethargy in dogs.",
    "Ehrlichiosis": "A tick-borne disease affecting white blood cells, causing fever and bleeding disorders.",
    "Rocky mountain spotted fever": "A bacterial disease transmitted by ticks, leading to fever and joint pain.",
    "Clostridium": "A bacterial infection causing severe diarrhea and gastrointestinal distress.",
    "Kennel cough": "A contagious respiratory infection in dogs, leading to a persistent dry cough.",
    "Blastomycosis": "A fungal infection that affects the lungs and can spread to other organs.",
    "Histoplasmosis": "A fungal infection that starts in the lungs and can affect multiple organs.",
    "Coccidioidomycosis": "A fungal disease (Valley Fever) causing respiratory distress and fever in dogs.",
    "Cryptococcosis": "A fungal infection that affects the respiratory system and the nervous system.",
    "Ring worm": "A fungal skin infection causing circular patches of hair loss and itching.",
    "Aspergillosis": "A fungal infection affecting the respiratory system and sometimes spreading to organs.",
    "Pythiosis": "A rare but aggressive fungal-like infection affecting the skin and gastrointestinal tract.",
    "Mucormycosis": "A fungal infection affecting the skin and respiratory system, often seen in immunocompromised dogs.",
    "Glardiasis": "A parasitic infection causing diarrhea and gastrointestinal discomfort.",
    "Coccidiosis": "A parasitic infection in dogs leading to diarrhea and weight loss.",
    "Protothecosis": "A rare algal infection affecting the skin and organs in dogs.",
    "Trichinosis": "A parasitic infection affecting the muscles, leading to pain and fever.",
    "Echinococcosis": "A parasitic infection from tapeworms, causing cysts in the liver and lungs.",
    "HeartWorm": "A serious parasitic infection where worms grow in the heart and lungs, causing breathing issues.",
    "Panosteitis": "A bone disease in young dogs, causing pain and limping, often called 'growing pains'.",
    "Luxating Patella": "A knee condition where the kneecap dislocates, causing limping and discomfort."
}

# Streamlit UI
st.title("🐶 Animal Disease & Danger Prediction 🏥")

# Input fields
animal = st.selectbox("Select Animal", label_encoders["Name"].classes_)
symptoms = [
    st.selectbox(f"Select Symptom {i+1}", label_encoders[f"Sym{i+1}"].classes_)
    for i in range(5)
]

# Prediction Function
def predict_disease_and_danger(animal, symptoms):
    try:
        input_data = [label_encoders["Name"].transform([animal])[0]]
        for i in range(5):
            if symptoms[i] in label_encoders[f"Sym{i+1}"].classes_:
                input_data.append(label_encoders[f"Sym{i+1}"].transform([symptoms[i]])[0])
            else:
                input_data.append(-1)  # Unknown symptom handling

        input_data = np.array(input_data).reshape(1, -1)  # Reshape for prediction

        # Predict Disease & Danger
        disease_pred = disease_model.predict(input_data)[0]
        danger_pred = danger_model.predict(input_data)[0]

        disease = label_encoders["Disease"].inverse_transform([disease_pred])[0].strip()
        danger = label_encoders["Danger"].inverse_transform([danger_pred])[0].strip()

        # Get disease explanation
        explanation = disease_info.get(disease, "Description not found.")
        return f"🦠 **Predicted Disease:** {disease}\n📌 **Explanation:** {explanation}\n⚠️ **Danger Level:** {danger}"
    
    except ValueError as e:
        return f"⚠️ Error: {e}. Please check input values."

# Predict Button
if st.button("Predict Disease & Danger"):
    result = predict_disease_and_danger(animal, symptoms)
    for line in result.split("\n"):  # Split result into separate lines
        st.write(line)

