from flask import Flask, request, jsonify, send_from_directory, abort, session
from flask_cors import CORS
import pickle
import numpy as np
import os
from PIL import Image
import logging
import requests
from io import BytesIO
import joblib
from chatbot_core import ChatBotCore
from dotenv import load_dotenv
import uuid
from Predict_db import save_disease_prediction, save_crop_recommendation, save_fertilizer_recommendation

load_dotenv()
#ChatBot::::___
chatbots = {}


def get_chatbot():
    """Get or create a chatbot instance for the current session"""
    # Get or create a session ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        
    session_id = session['session_id']
    
    # Create a new bot if needed
    if session_id not in chatbots:
        chatbots[session_id] = ChatBotCore()
    
    return chatbots[session_id]



# ======================
# INITIALIZATION
# ======================
app = Flask(__name__, static_folder='static')
CORS(app)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# ======================
# CONFIGURATION
# ======================
TF_SERVING_ENDPOINT = "http://localhost:8501/v1/models/potatoes_model:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
INPUT_SHAPE = (256, 256)
CONFIDENCE_THRESHOLD = 0.7
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Logging Config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================    
# MODEL LOADING
# ======================
# Load crop recommendation model
crop_model = pickle.load(open('model.pkl', 'rb'))
crop_model = pickle.load(open('model.pkl', 'rb'))
fert_model = joblib.load('fertilizer_model.pkl')
soil_encoder = joblib.load('soil_encoder.pkl')
crop_encoder = joblib.load('crop_encoder.pkl')
fertilizer_encoder = joblib.load('fertilizer_encoder.pkl')


with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)





# ======================
# ERROR HANDLERS
# ======================
@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(500)
@app.errorhandler(413)
def handle_error(error):
    code = getattr(error, 'code', 500)
    description = getattr(error, 'description', str(error))
    
    response = jsonify({
        "error": description,
        "status": "error",
        "code": code
    })
    response.status_code = code
    return response

# ======================
# IMAGE PROCESSING (For Disease Prediction)
# ======================
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(INPUT_SHAPE)
        return np.array(img)
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        abort(400, description=f"Invalid image: {str(e)}")

def is_potato_leaf(image_array: np.ndarray) -> bool:
    try:
        hsv = Image.fromarray(image_array).convert('HSV')
        h, s, v = hsv.split()
        green_pixels = np.sum((np.array(h) > 30) & (np.array(h) < 90))
        green_ratio = green_pixels / (INPUT_SHAPE[0] * INPUT_SHAPE[1])
        return green_ratio > 0.3
    except Exception as e:
        logger.error(f"Potato leaf validation failed: {str(e)}")
        return False

# ======================
# API ENDPOINTS
# ======================

# 🔸 Root route - serves landing page
@app.route('/')
def home():
    return send_from_directory('static', 'landingpage.html')

# 🔸 Crop Recommendation
@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    try:
        data = request.get_json()
        features = [
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]
        final_input = np.array([features])
        prediction = crop_model.predict(final_input)[0]
        
        session_id = session.get('session_id', str(uuid.uuid4()))
        save_crop_recommendation(
            session_id=session_id,
            recommended_crop=prediction,
            N=data['N'],
            P=data['P'],
            K=data['K'],
            temperature=data['temperature'],
            humidity=data['humidity'],
            ph=data['ph'],
            rainfall=data['rainfall']
        )
    

        return jsonify({'recommended_crop': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

# 🔸 Fertilizer Recommendation
# 🔸 Fertilizer Recommendation
@app.route('/predict-fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        data = request.get_json()

        required_fields = ['Temperature', 'Humidity', 'Moisture',
                           'Soil Type', 'Crop Type', 'Nitrogen',
                           'Phosphorous', 'Potassium']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Encode categorical inputs
        soil_encoded = soil_encoder.transform([data['Soil Type']])[0]
        crop_encoded = crop_encoder.transform([data['Crop Type']])[0]

        features = np.array([[data['Temperature'], data['Humidity'], data['Moisture'],
                              soil_encoded, crop_encoded,
                              data['Nitrogen'], data['Phosphorous'], data['Potassium']]])

        prediction_encoded = fert_model.predict(features)[0]
        fertilizer_name = fertilizer_encoder.inverse_transform([prediction_encoded])[0]

        return jsonify({'recommended_fertilizer': fertilizer_name})

    except ValueError as ve:
        return jsonify({'error': f'Encoding error: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Fertilizer prediction failed: {str(e)}'}), 500
# 🔸 Disease Prediction
@app.route('/predict', methods=['POST'])  # Match the URL your frontend uses
def predict():
    if 'file' not in request.files:
        abort(400, description="No file part in the request")

    file = request.files['file']
    if file.filename == '':
        abort(400, description="No file selected")

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        abort(400, description="Only PNG/JPG/JPEG images are allowed")

    try:
        # Read image file
        image_bytes = file.read()
        image = preprocess_image(image_bytes)
        
        # Validate potato leaf
        if not is_potato_leaf(image):
            abort(400, description="The image doesn't appear to be a potato leaf. Please upload a clear image of potato leaves.")

        # Prepare prediction request
        img_batch = np.expand_dims(image, axis=0)
        json_data = {"instances": img_batch.tolist()}

        # Call TensorFlow Serving
        response = requests.post(TF_SERVING_ENDPOINT, json=json_data)
        response.raise_for_status()
        
        # Process predictions
        predictions = response.json()['predictions'][0]
        logger.info(f"Raw model output: {predictions}")

        probs = np.array(predictions)
        if np.sum(probs) > 1.0 + 1e-5:  # Account for floating point errors
            probs = probs / np.sum(probs)

        predicted_idx = np.argmax(probs)
        confidence = float(probs[predicted_idx])

        # Prepare response
        response_data = {
            "prediction": CLASS_NAMES[predicted_idx],
            "confidence": round(confidence * 100, 2),
            "all_scores": {cls: float(prob*100) for cls, prob in zip(CLASS_NAMES, probs)}
        }

        if confidence < CONFIDENCE_THRESHOLD:
            response_data.update({
                "status": "low_confidence",
                "message": "Model isn't confident about this prediction"
            })
        else:
            response_data["status"] = "success"
            try:
                session_id = session.get('session_id', str(uuid.uuid4()))
                save_disease_prediction(
                    session_id=session_id,
                    prediction=CLASS_NAMES[predicted_idx],
                    confidence=round(confidence * 100, 2),
                    image_path=file.filename if file else None
                )
            except Exception as e:
                logger.error(f"Database insertion failed: {str(e)}")
                response_data["database_status"] = "failed"
                response_data["database_error"] = str(e)
        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        logger.error(f"TF Serving request failed: {str(e)}")
        abort(503, description="Model server unavailable")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        abort(500, description=str(e))


#AI_Chat bot :=>
@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a text chat message"""
    chatbot = get_chatbot()
    data = request.json
    user_input = data.get('message', '')
    model = data.get('model', 'llama3-8b-8192')
    memory_length = data.get('memory_length', 5)
    use_tts = data.get('use_tts', True)
    
    prompt_mode = data.get('prompt_mode', 'farmer')

    if prompt_mode == 'farmer':
        system_prompt = (
            "You are KrushiGPT, an AI assistant designed to help Indian farmers. "
                "Provide accurate, simple, and practical advice only on agriculture, fertilizers, crop diseases, "
                "and weather in Marathi or simple English.\n\n"
               "STRICT RESPONSE RULES:"
                "1. Each bullet on NEW LINE with BLANK LINE before it"
                "2. Use VARIABLE EMOJIS based on content:"
                    "- Crops: 🌾(wheat), 🌱(seedling), 🍚(rice), 🥜(groundnut)"
                    "- Actions: 💧(water), ✂️(prune), 🌿(organic), 🧪(chemical)"
                    "- Problems: 🐛(pest), 🦠(disease), ⚠️(warning), 🔥(blight)"
                    "- Weather: ☀️(sun), 🌧️(rain), 🌪️(storm), ❄️(frost)"
                "3. Never repeat same emoji consecutively"
                "4. Match emoji to Marathi/English content"
        )
    else:
        system_prompt = (
            "You are a general-purpose helpful assistant. Be friendly and informative."
        )

    full_prompt = f"{system_prompt}\nUser: {user_input}"

    # Process the message
    response_text = chatbot.process_message(full_prompt, model, memory_length, speak=use_tts)
    
    result = {
        'response': response_text,
        'chatHistory': chatbot.chat_history,  # Return chat history
        'speech_enabled': use_tts
    }
    
    return jsonify(result)

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech using the chatbot's TTS engine"""
    chatbot = get_chatbot()
    data = request.json
    text = data.get('text', '')
    
    # Use the chatbot's TTS function
    chatbot.text_to_speech(text)
    
    return jsonify({
        'status': 'success',
        'message': 'TTS processing started'
    })

@app.route('/api/stop-tts', methods=['POST'])
def stop_tts():
    """Stop ongoing TTS playback"""
    chatbot = get_chatbot()
    # Ensure speech is stopped immediately
    chatbot.stop_speaking()
    
    return jsonify({
        'status': 'success',
        'message': 'Speech stopped'
    })

@app.route('/api/voice', methods=['POST'])
def voice_input():
    """Process voice input from the frontend"""
    chatbot = get_chatbot()
    data = request.json
    text = data.get('text', '')
    model = data.get('model', 'llama3-8b-8192')
    memory_length = data.get('memory_length', 5)
    use_tts_response = data.get('use_tts_response', True)
    
    prompt_mode = data.get('prompt_mode', 'farmer')

    if prompt_mode == 'farmer':
        system_prompt = (
            "You are **KrushiGPT**, an AI assistant designed to help Indian farmers.\n\n"
            "🧠 Your job: Provide accurate, simple, and practical advice only on:\n"
            "- Agriculture\n"
            "- Fertilizers\n"
            "- Crop diseases\n"
            "- Weather\n"
            "Reply in **Marathi** or **simple English** depending on the user input.\n\n"
            
            "📌 **STRICT RESPONSE RULES**:\n\n"
            
            "1️⃣ Each bullet point should be on a **new line**, with a **blank line before it**.\n\n"
            
            "2️⃣ Use **appropriate emojis** based on content:\n"
            "- 🌾 (wheat), 🌱 (seedling), 🍚 (rice), 🥜 (groundnut)\n"
            "- 💧 (water), ✂️ (prune), 🌿 (organic), 🧪 (chemical)\n"
            "- 🐛 (pest), 🦠 (disease), ⚠️ (warning), 🔥 (blight)\n"
            "- ☀️ (sun), 🌧️ (rain), 🌪️ (storm), ❄️ (frost)\n\n"
            
            "3️⃣ **Never repeat the same emoji consecutively.**\n\n"
            
            "4️⃣ Match the emoji to the meaning of the content (English or Marathi).\n\n"
            
            "✅ Always be concise, direct, and helpful. Avoid unnecessary text."

            "Example:\n"
            "🌱 **Crop:** Tomato\n\n"
            "🦠 **Problem:** Early blight with brown leaf spots\n\n"
            "🧪 **Treatment:** Use Mancozeb (2 gm/litre) and spray every 7 days\n\n"
            "⚠️ **Tip:** Remove infected leaves to prevent spread"

        )

    else:
        system_prompt = (
            "You are a general-purpose helpful assistant. Be friendly and informative."
        )

    full_prompt = f"{system_prompt}\nUser: {text}"

    # Process the message
    response_text = chatbot.process_message(text, model, memory_length, speak=use_tts_response)
    
    result = {
        'text': text,
        'response': response_text,
        'speech_enabled': use_tts_response
    }
    
    return jsonify(result)


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update chatbot settings (TTS preferences)"""
    chatbot = get_chatbot()
    data = request.json
    always_speak = data.get('always_speak', None)
    
    if always_speak is not None:
        chatbot.always_speak = always_speak
    
    return jsonify({
        'status': 'success',
        'settings': {
            'always_speak': chatbot.always_speak
        }
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the conversation and stop any ongoing speech"""
    chatbot = get_chatbot()
    data = request.json
    model = data.get('model', 'llama3-8b-8192')
    memory_length = data.get('memory_length', 5)
    
    # Reset the conversation (this will also stop any ongoing speech)
    chatbot.initialize_conversation(model, memory_length)
    
    return jsonify({
        'status': 'success',
        'message': 'Conversation reset successfully'
    })

@app.route('/api/listen', methods=['POST'])
def listen():
    """Activate speech recognition and return transcribed text"""
    chatbot = get_chatbot()
    
    # Use the chatbot's speech recognition function
    transcribed_text = chatbot.speech_to_text()
    
    if not transcribed_text:
        return jsonify({
            'status': 'error',
            'message': 'No speech detected or could not transcribe audio'
        }), 400
    
    return jsonify({
        'status': 'success',
        'text': transcribed_text
    })


@app.route('/api/stop-speaking', methods=['POST'])
def stop_speaking_endpoint():
    """Stop the chatbot from speaking without resetting the conversation"""
    chatbot = get_chatbot()
    # Stop the speech immediately
    chatbot.stop_speaking()
    
    return jsonify({
        "status": "success",
        "message": "Speech stopped successfully"
    })

# ======================
# STATIC FILES
# ======================
@app.route('/detect')
def serve_detection():
    return send_from_directory('static', 'DiseasePrediction.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# Add this route with your other static file routes
@app.route('/crop-recommendation')
def serve_crop_recommendation():
    return send_from_directory('static', 'crop_recommendation.html')
# Add this route with your other static file routes
@app.route('/fertilizer-recommendation')
def serve_fertilizer_recommendation():
    return send_from_directory('static', 'fertilizer_recommendation2.html')

@app.route('/ai-assistant')
def serve_ai_assistant():
    return send_from_directory('static', 'ai_assistant.html')

@app.route('/subsidy-finder')
def serve_subsidy_finder():
    return send_from_directory('static', 'subsidy.html')






# ======================
# MAIN
# ======================
if __name__ == '__main__':
    app.run(host='localhost', port=2025, debug=True)