import streamlit as st
import os
import io
import base64
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
        st.stop()
    return OpenAI(api_key=api_key)

# Memory-based image processing with enhancement
class InMemoryImageProcessor:
    """Process images directly in memory with OCR optimization"""
    
    @staticmethod
    def process_uploaded_file(uploaded_file):
        """Process uploaded file directly in memory"""
        try:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            # Open image directly from bytes
            image = Image.open(uploaded_file)
            return image
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    
    @staticmethod
    def enhance_for_ocr(image):
        """Enhance image quality for better OCR accuracy"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Enhance brightness slightly
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            return image
        except Exception as e:
            st.warning(f"Image enhancement failed: {e}")
            return image

# Initialize image processor
@st.cache_resource
def get_image_processor():
    return InMemoryImageProcessor()

def encode_image_to_base64(uploaded_file):
    """Convert uploaded file to base64 for OpenAI API"""
    try:
        uploaded_file.seek(0)
        return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None

def extract_text_with_gpt4o_mini(client, uploaded_file):
    """Extract text from image using GPT-4o-mini (cost-effective OCR)"""
    base64_image = encode_image_to_base64(uploaded_file)
    
    if not base64_image:
        return None
    
    # Simple OCR prompt - much cheaper than full analysis
    ocr_prompt = """Extract ALL visible text from this food package image.

Focus on:
- Ingredient lists (in any language)
- Product name
- Nutritional information
- Any text on labels

Return ONLY the extracted text, preserving the original language and formatting. Do not analyze or translate - just extract text exactly as written."""

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ocr_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,  # Sufficient for text extraction
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OCR extraction failed: {e}")
        return None

def analyze_text_with_gpt35(client, extracted_text):
    """Analyze extracted text using GPT-3.5-turbo with JAKIM halal standards"""
    
    analysis_prompt = f"""You are a halal food analyst following JAKIM (Jabatan Kemajuan Islam Malaysia) halal standards. Analyze the following extracted text from a food package for halal compliance according to JAKIM guidelines.

Extracted text from package:
{extracted_text}

CRITICAL INSTRUCTIONS:
1. You MUST analyze ALL ingredients found in the text - do not skip any
2. You MUST translate ALL ingredient names to proper English
3. Follow JAKIM halal standards strictly
4. Respond with ONLY valid JSON - no explanations, no markdown
5. Ensure every ingredient has a clear English translation

OUTPUT FORMAT - EXACT JSON STRUCTURE:
{{
  "ingredients": [
    {{
      "name": "proper english translation of ingredient",
      "original_text": "original text from package",
      "halal_status": "halal",
      "reason": "brief reason based on JAKIM standards"
    }}
  ],
  "overall_halal_confidence": 85,
  "main_concerns": "brief summary of main issues per JAKIM",
  "nutritional_info": {{
    "calories_per_100g": 350,
    "protein_g": 12.5,
    "fat_g": 15.0,
    "carbs_g": 45.0,
    "fiber_g": 3.0,
    "sodium_mg": 500,
    "sugar_g": 8.0,
    "saturated_fat_g": 4.5
  }}
}}

JAKIM HALAL STANDARDS - KEY FOCUS AREAS:
1. **Animal-derived ingredients**: Must be from halal animals slaughtered according to Islamic law
2. **Alcohol**: Completely prohibited, including as processing aid or residual amounts
3. **Gelatin**: Only halal if from halal animal sources or fish gelatin
4. **Emulsifiers (E471, E472)**: Haram if from pork, halal if from halal animals/plants
5. **Lecithin**: Halal if from soy/sunflower, haram if from egg (if from non-halal source)
6. **Enzymes**: Must be from halal sources (microbial/plant preferred)
7. **Flavoring**: Natural flavors acceptable, artificial flavors need verification
8. **Mono- and Diglycerides**: Haram if from pork fat, halal if from plants/halal animals
9. **Whey**: Halal only if from halal-certified rennet
10. **Processing aids**: All processing aids must be halal-compliant

JAKIM E-NUMBER CLASSIFICATIONS:
- E120 (Carmine): HARAM - from insects
- E441 (Gelatin): SYUBHAH/HARAM unless verified halal source
- E471, E472: SYUBHAH unless certified halal source
- E631, E635: SYUBHAH - may contain pork-derived ingredients
- E920 (L-Cysteine): HARAM if from human hair, halal if synthetic

TRANSLATION EXAMPLES:
- "Zucker" → "Sugar"
- "Lecithine" → "Lecithin"
- "Emulgatoren" → "Emulsifiers"
- "Aromastoffe" → "Flavorings"
- "Gelatine" → "Gelatin"
- "Mono- und Diglyceride" → "Mono- and Diglycerides"

HALAL ASSESSMENT RULES (JAKIM Standards):
- halal_status must be exactly: "halal", "haram", "syubhah"
- Be STRICT: When in doubt about source, mark as "syubhah"
- overall_halal_confidence: integer 0-100 based on JAKIM compliance
- Mark as HARAM: Pork, alcohol, non-halal animal derivatives, prohibited E-numbers
- Mark as SYUBHAH: Unclear sources, questionable processing aids, dubious E-numbers

IMPORTANT: 
- Apply JAKIM standards strictly - be conservative in assessment
- Translate EVERY ingredient name to proper English
- Analyze EVERY ingredient mentioned in the text
- When unsure about halal status, always choose the more conservative option
- Provide comprehensive nutritional analysis including essential nutrients
- Respond with ONLY the JSON object above"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a JAKIM-certified halal food analyst and translator. You must apply JAKIM (Malaysia) halal standards strictly and translate ALL ingredients to proper English. Use conservative assessment - when in doubt, mark as syubhah. JAKIM standards are among the most stringent globally."
                },
                {
                    "role": "user", 
                    "content": analysis_prompt
                }
            ],
            max_tokens=1500,
            temperature=0.0,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None

def analyze_text_with_gpt4o_mini_fallback(client, extracted_text):
    """Fallback analysis using GPT-4o-mini with JAKIM halal standards"""
    
    analysis_prompt = f"""Analyze ALL ingredients from this food package text for halal compliance according to JAKIM (Malaysia) halal standards.

EXTRACTED TEXT:
{extracted_text}

You MUST respond with ONLY valid JSON covering ALL ingredients with proper English translations following JAKIM standards:

{{
  "ingredients": [
    {{
      "name": "proper english translation",
      "original_text": "original text from package",
      "halal_status": "halal",
      "reason": "brief explanation per JAKIM standards"
    }}
  ],
  "overall_halal_confidence": 85,
  "main_concerns": "summary per JAKIM guidelines",
  "nutritional_info": {{
    "calories_per_100g": 350,
    "protein_g": 12.5,
    "fat_g": 15.0,
    "carbs_g": 45.0,
    "fiber_g": 3.0,
    "sodium_mg": 500,
    "sugar_g": 8.0,
    "saturated_fat_g": 4.5
  }}
}}

JAKIM HALAL STANDARDS - STRICT COMPLIANCE:
1. **Zero tolerance for alcohol** - even trace amounts
2. **Gelatin must be halal-certified** or from fish
3. **All emulsifiers** must have halal certification
4. **Animal enzymes** must be from halal sources
5. **Processing equipment** must not be contaminated with haram substances
6. **E-numbers** strictly evaluated per JAKIM approved list

CRITICAL JAKIM ASSESSMENT CRITERIA:
- Mark HARAM: Any pork derivatives, alcohol, non-halal animal products
- Mark SYUBHAH: Unclear sources, questionable E-numbers, uncertified ingredients
- Mark HALAL: Plant-based, clearly permissible, JAKIM-approved ingredients
- BE CONSERVATIVE: When source unclear, always mark as syubhah

TRANSLATION REQUIREMENTS:
- Translate ALL ingredient names to proper English
- "Zucker" → "Sugar"
- "Weizenmehl" → "Wheat Flour"
- "Lecithine" → "Lecithin"
- "Emulgatoren" → "Emulsifiers"
- "Gelatine" → "Gelatin"

CRITICAL: 
- Apply JAKIM standards strictly - be conservative
- Include EVERY ingredient with English translation
- halal_status must be: "halal", "haram", "syubhah"
- Use whole numbers for confidence (0-100)
- No text before or after JSON"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a JAKIM-certified halal analyst and translator. Apply JAKIM (Malaysia) halal standards strictly - the most stringent global standards. When in doubt, mark as syubhah. Translate all ingredients to English."
                },
                {
                    "role": "user", 
                    "content": analysis_prompt
                }
            ],
            max_tokens=1200,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Fallback analysis failed: {e}")
        return None

def hybrid_analysis(client, uploaded_file):
    """Perform hybrid analysis with silent processing (no package weight needed)"""
    
    # Step 1: Extract text silently
    extracted_text = extract_text_with_gpt4o_mini(client, uploaded_file)
    
    if not extracted_text or len(extracted_text.strip()) < 10:
        return None
    
    # Step 2: Try analysis with GPT-3.5-turbo first
    analysis_response = analyze_text_with_gpt35(client, extracted_text)
    
    if analysis_response:
        response_clean = analysis_response.strip()
        if response_clean and response_clean.startswith('{') and len(response_clean) > 50:
            return analysis_response
    
    # Step 3: Fallback to GPT-4o-mini if needed
    analysis_response = analyze_text_with_gpt4o_mini_fallback(client, extracted_text)
    
    if analysis_response:
        response_clean = analysis_response.strip()
        if response_clean.startswith('{') and len(response_clean) > 50:
            return analysis_response
    
    return None

def parse_halal_response(response_text):
    """Parse and validate the halal analysis response with minimal user messaging"""
    
    if not response_text or not response_text.strip():
        return create_minimal_response("Empty AI response")
    
    try:
        response_text = response_text.strip()
        
        # Check if response is too short
        if len(response_text) < 50:
            return create_minimal_response("Response too short")
        
        # Handle truncation
        if not response_text.endswith('}'):
            if response_text.count('{') > response_text.count('}'):
                missing_braces = response_text.count('{') - response_text.count('}')
                response_text += '}' * missing_braces
        
        # Extract JSON
        json_str = response_text
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            if json_end != -1:
                json_str = response_text[json_start:json_end].strip()
            else:
                json_str = response_text[json_start:].strip()
        elif "```" in response_text:
            parts = response_text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("{") and "}" in part:
                    json_str = part
                    break
        
        # Find JSON boundaries
        if "{" in json_str:
            json_start = json_str.find("{")
            brace_count = 0
            json_end = len(json_str)
            
            for i in range(json_start, len(json_str)):
                if json_str[i] == "{":
                    brace_count += 1
                elif json_str[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            json_str = json_str[json_start:json_end]
        else:
            return create_minimal_response("No JSON found")
        
        # Handle incomplete JSON
        if not json_str.endswith("}"):
            if '"nutritional_info"' in json_str and json_str.count('"nutritional_info"') == 1:
                nutrition_start = json_str.find('"nutritional_info"')
                after_nutrition = json_str[nutrition_start:]
                
                if after_nutrition.count('{') > after_nutrition.count('}'):
                    json_str = json_str.rstrip(',') + '''
    },
    "nutritional_info": {
      "calories_per_100g": 0,
      "protein_g": 0,
      "fat_g": 0,
      "carbs_g": 0,
      "fiber_g": 0,
      "sodium_mg": 0,
      "sugar_g": 0,
      "saturated_fat_g": 0
    }
  }'''
            
            if not json_str.endswith('}'):
                json_str += '}'
        
        # Common JSON fixes
        json_str = json_str.replace("'", '"')
        json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        
        # Fix trailing commas
        import re
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return create_minimal_response("JSON parsing failed")
        
        # Validate and return
        return validate_and_fix_data(data)
        
    except Exception:
        return create_minimal_response("Parsing error")

def create_minimal_response(reason):
    """Create a minimal valid response when parsing fails"""
    return {
        "ingredients": [
            {
                "name": "Analysis incomplete",
                "original_text": "Could not parse AI response",
                "halal_status": "syubhah",
                "reason": reason
            }
        ],
        "overall_halal_confidence": 30,
        "main_concerns": f"Analysis failed: {reason}",
        "nutritional_info": {
            "calories_per_100g": 0,
            "protein_g": 0,
            "fat_g": 0,
            "carbs_g": 0,
            "fiber_g": 0,
            "sodium_mg": 0,
            "sugar_g": 0,
            "saturated_fat_g": 0
        }
    }

def validate_and_fix_data(data):
    """Validate and fix parsed JSON data"""
    # Ensure all required fields exist
    required_fields = ['ingredients', 'overall_halal_confidence', 'main_concerns', 'nutritional_info']
    for field in required_fields:
        if field not in data:
            if field == 'ingredients':
                data[field] = []
            elif field == 'overall_halal_confidence':
                data[field] = 50
            elif field == 'main_concerns':
                data[field] = "Incomplete analysis"
            elif field == 'nutritional_info':
                data[field] = {
                    "calories_per_100g": 0,
                    "protein_g": 0,
                    "fat_g": 0,
                    "carbs_g": 0,
                    "fiber_g": 0,
                    "sodium_mg": 0,
                    "sugar_g": 0,
                    "saturated_fat_g": 0
                }
    
    # Validate ingredients
    if not isinstance(data['ingredients'], list):
        data['ingredients'] = []
    
    if len(data['ingredients']) == 0:
        data['ingredients'] = [{
            "name": "No ingredients detected",
            "original_text": "Analysis incomplete",
            "halal_status": "syubhah",
            "reason": "Could not identify ingredients"
        }]
    
    # Fix each ingredient
    for i, ingredient in enumerate(data['ingredients']):
        if not isinstance(ingredient, dict):
            continue
        
        required_fields = ['name', 'halal_status', 'reason']
        for field in required_fields:
            if field not in ingredient:
                if field == 'name':
                    ingredient[field] = f"Ingredient {i+1}"
                elif field == 'halal_status':
                    ingredient[field] = "syubhah"
                elif field == 'reason':
                    ingredient[field] = "Incomplete data"
    
    # Fix confidence score
    try:
        confidence = float(data['overall_halal_confidence'])
        data['overall_halal_confidence'] = max(0, min(100, int(confidence)))
    except (ValueError, TypeError):
        data['overall_halal_confidence'] = 50
    
    # Ensure main_concerns is a string
    if not isinstance(data['main_concerns'], str):
        data['main_concerns'] = str(data['main_concerns']) if data['main_concerns'] else ""
    
    return data

def display_simplified_results(analysis):
    """Display simplified analysis results for mobile"""
    
    # 1. HALAL STATUS - Mobile-optimized confidence display
    confidence = analysis['overall_halal_confidence']
    
    # Large, prominent confidence with emoji
    if confidence >= 80:
        st.markdown("### ✅ HALAL")
        st.success(f"**{confidence}% Confidence**")
        st.success("✅ This product appears to be halal according to JAKIM Malaysia standards")
    elif confidence >= 60:
        st.markdown("### ⚠️ CAUTION")
        st.warning(f"**{confidence}% Confidence**")
        st.warning("⚠️ This product has some concerns under JAKIM standards - review carefully")
    else:
        st.markdown("### ❌ AVOID")
        st.error(f"**{confidence}% Confidence**")
        st.error("❌ This product has significant concerns under JAKIM standards")
    
    # Main concerns with mobile-friendly formatting
    if analysis['main_concerns'] and analysis['main_concerns'].strip():
        st.warning(f"⚠️ **Concerns:** {analysis['main_concerns']}")
    
    st.markdown("---")
    
    # 2. INGREDIENTS SUMMARY - Simplified bullet points
    st.markdown("### 📋 Ingredients Summary")
    
    # Count ingredients by status
    halal_count = sum(1 for ing in analysis['ingredients'] if ing['halal_status'].lower() == 'halal')
    haram_count = sum(1 for ing in analysis['ingredients'] if ing['halal_status'].lower() == 'haram')
    syubhah_count = sum(1 for ing in analysis['ingredients'] if ing['halal_status'].lower() == 'syubhah')
    total_count = len(analysis['ingredients'])
    
    # Simple bullet point summary
    st.markdown(f"""
    • **Total ingredients:** {total_count}
    • **✅ Halal:** {halal_count}
    • **⚠️ Syubhah:** {syubhah_count}
    • **❌ Haram:** {haram_count}
    """)
    
    st.markdown("---")
    
    # 3. NUTRITIONAL FACTS - Essential nutrients for healthy intake
    st.markdown("### 📊 Essential Nutrition Facts (per 100g)")
    
    nutrition = analysis['nutritional_info']
    
    # Display essential nutrients in a mobile-friendly grid
    col1, col2 = st.columns(2)
    
    with col1:
        # Energy and Macronutrients
        st.markdown("**Energy & Macros:**")
        calories = nutrition.get('calories_per_100g', 'N/A')
        protein = nutrition.get('protein_g', 'N/A')
        carbs = nutrition.get('carbs_g', 'N/A')
        fat = nutrition.get('fat_g', 'N/A')
        
        st.metric("🔥 Energy", f"{calories} kcal")
        st.metric("🥩 Protein", f"{protein}g")
        st.metric("🌾 Carbs", f"{carbs}g")
        st.metric("🫒 Total Fat", f"{fat}g")
    
    with col2:
        # Essential Micronutrients
        st.markdown("**Essential Nutrients:**")
        fiber = nutrition.get('fiber_g', 'N/A')
        sodium = nutrition.get('sodium_mg', 'N/A')
        sugar = nutrition.get('sugar_g', 'N/A')
        saturated_fat = nutrition.get('saturated_fat_g', 'N/A')
        
        st.metric("🌿 Fiber", f"{fiber}g")
        st.metric("🧂 Sodium", f"{sodium}mg")
        st.metric("🍯 Sugar", f"{sugar}g")
        st.metric("🔴 Sat. Fat", f"{saturated_fat}g")
    
    # Nutritionist's Assessment
    st.markdown("#### 👩‍⚕️ Nutritionist's Assessment")
    nutritional_assessment = generate_nutrition_assessment(nutrition)
    
    # Display assessment with appropriate color coding
    if nutritional_assessment['overall_rating'] == 'healthy':
        st.success(f"✅ **{nutritional_assessment['summary']}**")
    elif nutritional_assessment['overall_rating'] == 'moderate':
        st.warning(f"⚠️ **{nutritional_assessment['summary']}**")
    else:
        st.error(f"❌ **{nutritional_assessment['summary']}**")
    
    # Detailed nutritional insights
    with st.expander("📋 Detailed Nutritional Analysis"):
        for insight in nutritional_assessment['insights']:
            if insight['type'] == 'positive':
                st.success(f"✅ {insight['message']}")
            elif insight['type'] == 'warning':
                st.warning(f"⚠️ {insight['message']}")
            else:
                st.error(f"❌ {insight['message']}")
        
        # Recommendations
        if nutritional_assessment['recommendations']:
            st.markdown("**💡 Recommendations:**")
            for rec in nutritional_assessment['recommendations']:
                st.info(f"• {rec}")

def generate_nutrition_assessment(nutrition):
    """Generate nutritionist's assessment of the nutritional profile"""
    
    insights = []
    recommendations = []
    concerns = 0
    positives = 0
    
    # Get nutritional values (handle N/A values)
    def safe_float(value, default=0):
        try:
            return float(value) if value != 'N/A' else default
        except (ValueError, TypeError):
            return default
    
    calories = safe_float(nutrition.get('calories_per_100g'))
    protein = safe_float(nutrition.get('protein_g'))
    carbs = safe_float(nutrition.get('carbs_g'))
    fat = safe_float(nutrition.get('fat_g'))
    fiber = safe_float(nutrition.get('fiber_g'))
    sodium = safe_float(nutrition.get('sodium_mg'))
    sugar = safe_float(nutrition.get('sugar_g'))
    sat_fat = safe_float(nutrition.get('saturated_fat_g'))
    
    # Calorie Assessment (per 100g)
    if calories == 0:
        insights.append({"type": "warning", "message": "Nutritional information incomplete - unable to provide full assessment"})
    elif calories < 150:
        insights.append({"type": "positive", "message": "Low calorie density - good for weight management"})
        positives += 1
    elif calories > 400:
        insights.append({"type": "warning", "message": "High calorie density - consume in moderation"})
        concerns += 1
        recommendations.append("Consider smaller portion sizes due to high calorie content")
    
    # Protein Assessment
    if protein >= 10:
        insights.append({"type": "positive", "message": "Good protein content - supports muscle health and satiety"})
        positives += 1
    elif protein >= 5:
        insights.append({"type": "warning", "message": "Moderate protein content"})
    elif protein > 0:
        insights.append({"type": "warning", "message": "Low protein content - may not be very filling"})
        recommendations.append("Pair with protein-rich foods for balanced nutrition")
    
    # Fat Assessment
    if fat > 20:
        insights.append({"type": "warning", "message": "High fat content - check saturated fat levels"})
        concerns += 1
    elif fat >= 10:
        insights.append({"type": "warning", "message": "Moderate fat content"})
    elif fat < 3 and fat > 0:
        insights.append({"type": "positive", "message": "Low fat content - heart-friendly option"})
        positives += 1
    
    # Saturated Fat Assessment
    if sat_fat > 5:
        insights.append({"type": "negative", "message": "High saturated fat - limit intake for heart health"})
        concerns += 1
        recommendations.append("Limit consumption due to high saturated fat content")
    elif sat_fat > 2:
        insights.append({"type": "warning", "message": "Moderate saturated fat levels"})
    elif sat_fat <= 1 and sat_fat > 0:
        insights.append({"type": "positive", "message": "Low saturated fat - heart-healthy choice"})
        positives += 1
    
    # Sugar Assessment
    if sugar > 15:
        insights.append({"type": "negative", "message": "High sugar content - may cause blood sugar spikes"})
        concerns += 1
        recommendations.append("Consume in small portions due to high sugar content")
    elif sugar > 5:
        insights.append({"type": "warning", "message": "Moderate sugar content - monitor intake"})
    elif sugar <= 2 and sugar >= 0:
        insights.append({"type": "positive", "message": "Low sugar content - good for blood sugar control"})
        positives += 1
    
    # Sodium Assessment
    if sodium > 600:
        insights.append({"type": "negative", "message": "High sodium content - may affect blood pressure"})
        concerns += 1
        recommendations.append("Limit intake if you have high blood pressure")
    elif sodium > 300:
        insights.append({"type": "warning", "message": "Moderate sodium levels"})
    elif sodium <= 100 and sodium >= 0:
        insights.append({"type": "positive", "message": "Low sodium - good for heart health"})
        positives += 1
    
    # Fiber Assessment
    if fiber >= 6:
        insights.append({"type": "positive", "message": "High fiber content - excellent for digestive health"})
        positives += 1
    elif fiber >= 3:
        insights.append({"type": "positive", "message": "Good fiber content - supports digestion"})
        positives += 1
    elif fiber > 0:
        insights.append({"type": "warning", "message": "Low fiber content"})
        recommendations.append("Add high-fiber foods to your meal for better digestion")
    
    # Overall Assessment
    if concerns == 0 and positives >= 3:
        overall_rating = "healthy"
        summary = "Nutritionally balanced food - suitable for regular consumption"
    elif concerns <= 1 and positives >= 2:
        overall_rating = "moderate"
        summary = "Moderately healthy - can be part of a balanced diet"
    elif concerns >= 2 or (concerns >= 1 and positives == 0):
        overall_rating = "concerning"
        summary = "Nutritional concerns present - consume occasionally in small portions"
    else:
        overall_rating = "moderate"
        summary = "Average nutritional profile - monitor portion sizes"
    
    # Add general recommendations if no specific ones
    if not recommendations:
        if overall_rating == "healthy":
            recommendations.append("This food can be part of a regular healthy diet")
        elif overall_rating == "moderate":
            recommendations.append("Enjoy in moderation as part of a balanced diet")
        else:
            recommendations.append("Consider healthier alternatives for regular consumption")
    
    return {
        "overall_rating": overall_rating,
        "summary": summary,
        "insights": insights,
        "recommendations": recommendations
    }
    
    st.markdown("---")
    
    # 4. INGREDIENTS DETAIL LIST - Mobile-optimized with priority sorting
    st.markdown("### 📝 Ingredients Detail List")
    st.caption("*Haram and Syubhah ingredients shown first*")
    
    # Sort ingredients by priority: haram -> syubhah -> halal
    def get_priority(ingredient):
        status = ingredient['halal_status'].lower()
        if status == 'haram':
            return 1  # Highest priority
        elif status == 'syubhah':
            return 2  # Medium priority
        else:  # halal
            return 3  # Lowest priority
    
    # Sort ingredients by priority
    sorted_ingredients = sorted(analysis['ingredients'], key=get_priority)
    
    for i, ingredient in enumerate(sorted_ingredients, 1):
        status = ingredient['halal_status'].lower()
        
        # Status icons and colors
        if status == "halal":
            icon = "✅"
            status_text = "HALAL"
        elif status == "haram":
            icon = "❌"
            status_text = "HARAM"
        else:
            icon = "⚠️"
            status_text = "SYUBHAH"
        
        # Get English name - ensure it's properly translated
        english_name = ingredient.get('name', 'Unknown ingredient')
        
        # Mobile-friendly expandable ingredient details with English name in title
        with st.expander(f"{icon} **{status_text}** - {english_name}", expanded=False):
            st.write(f"**English Name:** {english_name}")
            if 'original_text' in ingredient and ingredient['original_text'] and ingredient['original_text'] != ingredient['name']:
                st.write(f"**Original Text:** {ingredient['original_text']}")
            st.write(f"**Halal Status:** {ingredient['halal_status'].title()}")
            st.write(f"**Reason:** {ingredient['reason']}")
    
    st.markdown("---")
    
    # Simplified download section
    st.markdown("### 💾 Save Results")
    
    if st.button("📱 **DOWNLOAD REPORT**", 
                type="primary", 
                use_container_width=True,
                key="download_btn"):
        with st.spinner("📱 Creating report..."):
            img_buffer = create_simplified_report(analysis)
            if img_buffer:
                st.success("✅ Report ready!")
                st.download_button(
                    label="📥 **DOWNLOAD IMAGE**",
                    data=img_buffer,
                    file_name=f"halal_report_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                    mime="image/png",
                    use_container_width=True,
                    key="download_file_btn"
                )
                st.info("💡 Image saved to your phone's Downloads folder")
            else:
                st.error("❌ Could not create report. Please try again.")

def create_simplified_report(analysis):
    """Create simplified mobile-optimized result image"""
    try:
        # Mobile-optimized dimensions (portrait orientation)
        img_width = 600
        img_height = 800
        
        # Create white background
        result_img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(result_img)
        
        # Use default font (more reliable across platforms)
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            header_font = ImageFont.truetype("arial.ttf", 18)
            text_font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = header_font = text_font = ImageFont.load_default()
        
        y_pos = 20
        
        # Mobile-optimized header
        draw.rectangle([0, 0, img_width, 60], fill='#2E7D32')
        draw.text((20, 20), "🥘 JAKIM Halal Analysis Report", fill='white', font=title_font)
        y_pos = 80
        
        # Date
        draw.text((20, y_pos), f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fill='gray', font=text_font)
        y_pos += 35
        
        # Confidence score
        confidence = analysis['overall_halal_confidence']
        if confidence >= 80:
            bg_color = '#4CAF50'
            status_text = f"✅ {confidence}% HALAL"
        elif confidence >= 60:
            bg_color = '#FF9800'
            status_text = f"⚠️ {confidence}% CAUTION"
        else:
            bg_color = '#F44336'
            status_text = f"❌ {confidence}% AVOID"
        
        draw.rectangle([20, y_pos, img_width-20, y_pos + 50], fill=bg_color)
        draw.text((30, y_pos + 15), status_text, fill='white', font=header_font)
        y_pos += 70
        
        # Ingredients summary
        halal_count = sum(1 for ing in analysis['ingredients'] if ing['halal_status'].lower() == 'halal')
        haram_count = sum(1 for ing in analysis['ingredients'] if ing['halal_status'].lower() == 'haram')
        syubhah_count = sum(1 for ing in analysis['ingredients'] if ing['halal_status'].lower() == 'syubhah')
        total_count = len(analysis['ingredients'])
        
        draw.text((20, y_pos), f"📋 Total: {total_count} | Halal: {halal_count} | Syubhah: {syubhah_count} | Haram: {haram_count}", fill='black', font=text_font)
        y_pos += 35
        
        # Essential nutrition info (comprehensive)
        nutrition = analysis['nutritional_info']
        calories = nutrition.get('calories_per_100g', 'N/A')
        protein = nutrition.get('protein_g', 'N/A')
        fat = nutrition.get('fat_g', 'N/A')
        carbs = nutrition.get('carbs_g', 'N/A')
        fiber = nutrition.get('fiber_g', 'N/A')
        sodium = nutrition.get('sodium_mg', 'N/A')
        sugar = nutrition.get('sugar_g', 'N/A')
        sat_fat = nutrition.get('saturated_fat_g', 'N/A')
        
        draw.text((20, y_pos), "📊 Essential Nutrition (per 100g):", fill='black', font=header_font)
        y_pos += 25
        
        # Display key nutrients in compact format
        nutrition_line1 = f"Energy: {calories}kcal | Protein: {protein}g | Carbs: {carbs}g | Fat: {fat}g"
        nutrition_line2 = f"Fiber: {fiber}g | Sugar: {sugar}g | Sodium: {sodium}mg | Sat.Fat: {sat_fat}g"
        
        draw.text((20, y_pos), nutrition_line1, fill='black', font=text_font)
        y_pos += 20
        draw.text((20, y_pos), nutrition_line2, fill='black', font=text_font)
        y_pos += 35
        
        # Ingredients list with English names - prioritized by concern level
        draw.text((20, y_pos), "📝 Ingredients (⚠️ Concerns first):", fill='black', font=header_font)
        y_pos += 30
        
        # Sort ingredients by priority: haram -> syubhah -> halal
        def get_priority(ingredient):
            status = ingredient['halal_status'].lower()
            if status == 'haram':
                return 1  # Highest priority
            elif status == 'syubhah':
                return 2  # Medium priority
            else:  # halal
                return 3  # Lowest priority
        
        # Sort ingredients by priority
        sorted_ingredients = sorted(analysis['ingredients'], key=get_priority)
        
        # List ingredients with English names prominently
        for i, ingredient in enumerate(sorted_ingredients[:15]):  # Limit for mobile
            status = ingredient['halal_status'].lower()
            icon = "✅" if status == "halal" else "❌" if status == "haram" else "⚠️"
            
            # Show English name prominently
            english_name = ingredient.get('name', 'Unknown ingredient')
            if len(english_name) > 35:
                english_name = english_name[:32] + "..."
            
            # Add original text if available and different (but smaller)
            display_text = english_name
            if 'original_text' in ingredient and ingredient['original_text'] and ingredient['original_text'] != ingredient['name']:
                original = ingredient['original_text']
                if len(original) > 15:
                    original = original[:12] + "..."
                display_text = f"{english_name} ({original})"
            
            # Truncate if still too long
            if len(display_text) > 50:
                display_text = display_text[:47] + "..."
            
            draw.text((20, y_pos), f"{icon} {display_text}", fill='black', font=text_font)
            y_pos += 22
        
        # Footer
        footer_y = img_height - 60
        draw.rectangle([0, footer_y, img_width, img_height], fill='#F5F5F5')
        draw.text((20, footer_y + 10), "Generated by JAKIM Halal Checker", fill='gray', font=text_font)
        draw.text((20, footer_y + 30), "Based on JAKIM Malaysia standards", fill='gray', font=text_font)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        result_img.save(img_buffer, format='PNG', quality=90, optimize=True)
        img_buffer.seek(0)
        
        return img_buffer
        
    except Exception as e:
        st.error(f"Error creating simplified report: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Halal Ingredient Tracker",
        page_icon="🥘",
        layout="centered",  # Better for mobile
        initial_sidebar_state="collapsed"  # Hide sidebar on mobile
    )
    
    # Mobile-optimized CSS
    st.markdown("""
    <style>
    /* Mobile-first responsive design */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* Large, touch-friendly buttons */
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
    
    /* File uploader styling for mobile */
    .uploadedFile {
        width: 100%;
    }
    
    /* Mobile-optimized metrics */
    .metric-container {
        background: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.25rem 0;
        text-align: center;
    }
    
    /* Touch-friendly expandable sections */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        padding: 0.75rem;
    }
    
    /* Mobile typography */
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.3rem !important; }
    
    /* Hide Streamlit branding for cleaner mobile experience */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Mobile-optimized title
    st.markdown("# 🥘 Halal Checker")
    st.markdown("*Quick halal verification using **JAKIM Malaysia** standards*")
    st.info("🇲🇾 **JAKIM Standards**: Using Malaysia's strict halal certification guidelines - recognized globally for their stringent requirements")
    
    # Initialize clients
    client = get_openai_client()
    image_processor = get_image_processor()
    
    # Mobile-optimized file upload section
    st.markdown("## 📸 Upload Food Package")
    
    # Enhanced file uploader for mobile compatibility
    uploaded_file = st.file_uploader(
        "Take photo or choose image",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="📱 Tip: Take a clear photo of the ingredients list",
        accept_multiple_files=False,
        key="mobile_uploader"
    )
    
    # Mobile-specific file upload instructions
    if not uploaded_file:
        st.info("📱 **On mobile**: Tap 'Browse files' then choose 'Camera' to take a photo, or 'Photo Library' to select an existing image")
    
    # Check file size with mobile-friendly message
    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > 10:
            st.error(f"❌ Image too large ({file_size_mb:.1f}MB). Please use an image smaller than 10MB.")
            uploaded_file = None
        else:
            st.success(f"✅ Image uploaded ({file_size_mb:.1f}MB)")
    
    # Analysis section
    if uploaded_file is not None:
        # Large, prominent analyze button
        st.markdown("## 🔍 Analysis")
        if st.button("🔍 **ANALYZE INGREDIENTS**", 
                    type="primary", 
                    use_container_width=True,
                    key="analyze_btn"):
            
            with st.spinner("🔍 Analyzing ingredients..."):
                try:
                    # Process image
                    processed_image = image_processor.process_uploaded_file(uploaded_file)
                    
                    if processed_image is None:
                        st.error("❌ Could not process image. Please try a clearer photo.")
                        return
                    
                    # Perform analysis (no package weight needed)
                    response = hybrid_analysis(client, uploaded_file)
                    
                    if not response:
                        st.error("❌ Analysis failed. Please try with a clearer image of the ingredients list.")
                        return
                    
                    # Parse and display results
                    analysis = parse_halal_response(response)
                    
                    if analysis:
                        display_simplified_results(analysis)
                    else:
                        st.error("❌ Could not analyze ingredients. Please ensure the ingredients list is clearly visible.")
                        
                        # Show debug info for failed parsing
                        if response:
                            with st.expander("🔧 Debug: Raw AI Response", expanded=False):
                                st.code(response[:1000] + "..." if len(response) > 1000 else response)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    else:
        # Mobile-friendly upload instructions
        st.markdown("""
        ### 📱 How to use:
        1. **Tap 'Browse files'** above
        2. **Choose 'Camera'** to take a fresh photo
        3. **Point camera** at ingredients list
        4. **Take clear photo** with good lighting
        5. **Tap analyze** to get halal status
        
        #### 💡 Tips for best results:
        - Hold phone steady
        - Ensure good lighting
        - Make sure text is readable
        - Avoid glare and shadows
        """)

    # Footer
    st.markdown("---")
    st.markdown("*This tool applies **JAKIM Malaysia** halal standards. For official halal certification, consult JAKIM or authorized halal certification bodies.*")

if __name__ == "__main__":
    main()