# api/chat.py
"""
Voice Agent API for Vercel deployment
Handles conversation logic and LLM responses
"""

import json
import os
import tempfile
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Speech Recognition
try:
    import speech_recognition as sr
    GOOGLE_STT_AVAILABLE = True
except ImportError:
    GOOGLE_STT_AVAILABLE = False

@dataclass
class ConversationContext:
    """Real-time conversation context"""
    farmer_mood: str = "neutral"
    farmer_intent: str = "listening"
    concerns_raised: List[str] = None
    interest_signals: List[str] = None
    conversation_stage: str = "opening"
    
    def __post_init__(self):
        if self.concerns_raised is None:
            self.concerns_raised = []
        if self.interest_signals is None:
            self.interest_signals = []

class WebVoiceAgent:
    """Voice Agent optimized for web deployment"""
    
    def __init__(self):
        # Initialize Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and GEMINI_AVAILABLE:
            genai.configure(api_key=gemini_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.context = ConversationContext()
        else:
            self.model = None
        
        # Initialize STT
        if GOOGLE_STT_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
        else:
            self.recognizer = None
        
        # ElevenLabs setup
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = "EXAVITQu4vr4xnSDxMaL"  # Bella
    
    def interpret_farmer_response(self, farmer_message: str) -> ConversationContext:
        """Interpret farmer's response with LLM"""
        if not self.model:
            return self.context
        
        interpret_prompt = f"""QUICK ANALYSIS: Farmer said "{farmer_message}"

INTERPRET in 1-2 words each:
MOOD: [happy/confused/annoyed/interested/busy]
INTENT: [wants_info/has_concern/showing_interest/wants_to_end/asking_question]
STAGE: [opening/explaining/objection/closing/interested]

Format: MOOD:word INTENT:word STAGE:word"""

        try:
            response = self.model.generate_content(
                interpret_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=30,
                    temperature=0.3,
                )
            )
            
            analysis = response.text.strip()
            
            mood = "neutral"
            intent = "listening"
            stage = "explaining"
            
            for line in analysis.split():
                if "MOOD:" in line:
                    mood = line.split(":")[1]
                elif "INTENT:" in line:
                    intent = line.split(":")[1]
                elif "STAGE:" in line:
                    stage = line.split(":")[1]
            
            self.context.farmer_mood = mood
            self.context.farmer_intent = intent
            self.context.conversation_stage = stage
            
            return self.context
            
        except Exception as e:
            logger.error(f"Interpretation error: {e}")
            return self.context
    
    def generate_female_response(self, farmer_message: str, is_opening: bool = False) -> str:
        """Generate natural female Hinglish response"""
        if not self.model:
            return "Uncle ji, main PM-KUSUM scheme ke baare mein baat kar rahi thi. Government subsidy milti hai."
        
        if is_opening:
            prompt = """You are a friendly female agent calling an Indian farmer about PM-KUSUM solar pump scheme.

Create NATURAL FEMALE Hinglish opening (30-40 words max):
- Use warm, polite female tone with "ji", "uncle ji", "bhaiya"
- Mix Hindi-English naturally like educated Indian women speak
- Sound helpful and caring, not pushy
- Mention PM-KUSUM scheme with government backing
- Ask politely if they'd like to hear more

Examples:
- "Namaste uncle ji! Main PM-KUSUM scheme ke baare mein call kar rahi thi"
- "Government ka solar pump scheme hai - bahut achha subsidy mil raha hai"

Make it natural and caring."""
        else:
            # Update context
            self.context = self.interpret_farmer_response(farmer_message)
            
            mood_adaptations = {
                "confused": "patient and clear explanation like a helpful sister",
                "annoyed": "very polite and respectful tone", 
                "interested": "excited but professional female enthusiasm",
                "busy": "quick and respectful, understanding their time",
                "happy": "warm and encouraging female tone"
            }
            
            adaptation = mood_adaptations.get(self.context.farmer_mood, "natural female conversation")
            
            prompt = f"""Farmer said: "{farmer_message}"
Farmer seems: {self.context.farmer_mood}

Respond as a professional FEMALE agent in natural Hinglish (25-35 words):
- Use {adaptation}
- Mix Hindi-English like educated Indian women speak naturally
- Address farmer respectfully (uncle ji, bhaiya, sahab)
- Keep PM-KUSUM benefits simple and clear
- Sound caring and helpful, not salesy

Examples:
- "Haan uncle ji, bilkul samjh gaya! Yeh government scheme hai"
- "Dekho bhaiya, solar pump lagane se diesel ka paisa bach jata hai"
- "90% tak subsidy milti hai government se, sirf 10% aapko dena hai"

Keep it natural and caring."""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=80,
                    temperature=0.8,
                )
            )
            
            return response.text.strip()
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            fallbacks = [
                "Uncle ji, main PM-KUSUM solar scheme ke baare mein baat kar rahi thi. Government subsidy milti hai. Sunna chahenge?",
                "Bhaiya, solar pump se bijli ka bill kam ho jata hai. Interest hai aapko?",
                "Dekho ji, 90% government subsidy deti hai, bas 10% aapko dena hai. Achha deal hai na?"
            ]
            return fallbacks[0]
    
    def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio using Google STT"""
        if not self.recognizer:
            return None
        
        try:
            # Create temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.write(audio_data)
            temp_file.close()
            
            with sr.AudioFile(temp_file.name) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.record(source)
            
            # Transcribe with Indian English
            transcript = self.recognizer.recognize_google(audio, language='en-IN')
            
            # Cleanup
            os.unlink(temp_file.name)
            
            return transcript.strip() if transcript else None
            
        except Exception as e:
            logger.error(f"STT error: {e}")
            return None
    
    def generate_speech(self, text: str) -> Optional[bytes]:
        """Generate speech using ElevenLabs"""
        if not self.elevenlabs_key:
            return None
        
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_key
            }
            
            data = {
                "text": text[:200],  # Limit length
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {
                    "stability": 0.6,
                    "similarity_boost": 0.8,
                    "style": 0.4,
                    "use_speaker_boost": False
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.content
            else:
                return None
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

# Global agent instance
agent = WebVoiceAgent()

def handler(request):
    """Vercel serverless function handler"""
    
    # Handle CORS
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json'
    }
    
    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    if request.method != 'POST':
        return {
            'statusCode': 405,
            'headers': headers,
            'body': json.dumps({'error': 'Method not allowed'})
        }
    
    try:
        # Parse request
        if hasattr(request, 'get_json'):
            data = request.get_json()
        else:
            data = json.loads(request.body or '{}')
        
        action = data.get('action')
        
        if action == 'start':
            # Generate opening message
            response_text = agent.generate_female_response("", is_opening=True)
            audio_data = agent.generate_speech(response_text)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'text': response_text,
                    'audio': audio_data.hex() if audio_data else None,
                    'context': asdict(agent.context)
                })
            }
        
        elif action == 'text_response':
            # Handle text input
            farmer_message = data.get('message', '')
            response_text = agent.generate_female_response(farmer_message)
            audio_data = agent.generate_speech(response_text)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'text': response_text,
                    'audio': audio_data.hex() if audio_data else None,
                    'context': asdict(agent.context)
                })
            }
        
        elif action == 'voice_response':
            # Handle voice input
            audio_hex = data.get('audio')
            if not audio_hex:
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({'error': 'No audio data provided'})
                }
            
            # Convert hex to bytes
            audio_data = bytes.fromhex(audio_hex)
            
            # Transcribe
            transcript = agent.transcribe_audio(audio_data)
            
            if transcript:
                # Generate response
                response_text = agent.generate_female_response(transcript)
                response_audio = agent.generate_speech(response_text)
                
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({
                        'transcript': transcript,
                        'text': response_text,
                        'audio': response_audio.hex() if response_audio else None,
                        'context': asdict(agent.context)
                    })
                }
            else:
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({
                        'transcript': None,
                        'error': 'Could not understand audio'
                    })
                }
        
        else:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Invalid action'})
            }
    
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': str(e)})
        }