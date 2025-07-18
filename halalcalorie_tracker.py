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

# Image processing with enhancement
class ImageProcessor:
    """Process images directly in memory with OCR optimization"""
    
    @staticmethod
    def process_uploaded_file(uploaded_file):
        """Process uploaded file directly in memory"""
        try:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            return image
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    
    @staticmethod
    def enhance_for_ocr(image):
        """Enhance image quality for better OCR accuracy"""
        try:
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

@st.cache_resource
def get_image_processor():
    return ImageProcessor()

def encode_image_to_base64(uploaded_file):
    """Convert uploaded file to base64 for OpenAI API"""
    try:
        uploaded_file.seek(0)
        return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None

def extract_text_with_gpt4o_mini(client, uploaded_file):
    """Extract text from image using GPT-4o-mini"""
    base64_image = encode_image_to_base64(uploaded_file)
    
    if not base64_image:
        return None
    
    ocr_prompt = """Extract ALL visible text from this food package image.

Focus on:
- Ingredient lists (in any language)
- Product name
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
            max_tokens=500,
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OCR extraction failed: {e}")
        return None

def analyze_halal_status(client, extracted_text):
    """Analyze extracted text for halal status using JAKIM standards"""
    
    analysis_prompt = f"""You are a halal food analyst following JAKIM Malaysia halal standards. Analyze the extracted text for halal compliance.

Text: {extracted_text}

Respond with ONLY this JSON structure with NO additional text:

{{
  "ingredients": [
    {{
      "name": "English ingredient name",
      "original_text": "original text",
      "halal_status": "halal",
      "reason": "explanation"
    }}
  ],
  "overall_halal_status": "halal",
  "overall_halal_confidence": 85,
  "main_concerns": "summary"
}}

IMPORTANT RULES:
- halal_status MUST be exactly: "halal", "haram", or "syubhah"
- confidence MUST be integer 0-100
- NO markdown, NO code blocks, NO extra text
- ONLY the JSON object above

JAKIM MS 1500:2019 HALAL STANDARDS - CURRENT REQUIREMENTS:
1. **Animal-derived ingredients**: Must be from halal animals slaughtered according to Islamic law and Hukum Syarak
2. **Alcohol**: Completely prohibited - no tolerance for ethanol from fermented/distilled sources (exception: <0.1% if not from khamr)
3. **Gelatin**: Only halal if from halal-certified animal sources or fish gelatin (E441)
4. **Emulsifiers**: E470-E483 series HARAM if from pork or non-halal sources, halal if plant/halal animal derived
5. **Lecithin (E322)**: Halal if from soy/sunflower, SYUBHAH if from egg without halal certification
6. **Enzymes**: Must be from verified halal sources (microbial/plant preferred, animal enzymes require halal certification)
7. **Mono- and Diglycerides (E471, E472)**: HARAM if from pork fat, SYUBHAH without halal certification
8. **Whey**: Halal only if from halal-certified rennet
9. **Cross-contamination**: No mixing with non-halal sources or equipment contaminated with impure substances

JAKIM MS 1500:2019 E-NUMBER CLASSIFICATIONS (Updated 2025):
- E120 (Carmine/Cochineal): HARAM - derived from insects
- E422 (Glycerol/Glycerin): HARAM if from pork or non-halal meat sources
- E441 (Gelatin): HARAM unless from halal-certified sources or fish
- E470-E483 (Emulsifiers): HARAM if from pork or non-halal sources
- E471, E472 (Mono/Diglycerides): SYUBHAH unless halal-certified source
- E542 (Edible Bone Phosphate): HARAM if from pork or non-halal meat sources
- E631, E635 (Disodium/Calcium 5'-ribonucleotides): SYUBHAH - may contain pork derivatives
- E920 (L-Cysteine): HARAM if from human hair, halal if synthetic/plant-based
- E101 (Riboflavin): SYUBHAH if from pork liver/kidney, halal if 100% plant material

HALAL ASSESSMENT RULES (JAKIM Standards):
- overall_halal_status must be exactly: "halal", "haram", "syubhah"
- Be STRICT: When in doubt about source, mark as "syubhah"
- overall_halal_confidence: integer 0-100 based on JAKIM compliance
- Mark as HARAM: Pork, alcohol, non-halal animal derivatives, prohibited E-numbers
- Mark as SYUBHAH: Unclear sources, questionable processing aids, dubious E-numbers

Apply JAKIM standards strictly - be conservative in assessment. When unsure about halal status, always choose the more conservative option."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a JAKIM-certified halal food analyst. Apply JAKIM (Malaysia) halal standards strictly. Use conservative assessment - when in doubt, mark as syubhah."
                },
                {
                    "role": "user", 
                    "content": analysis_prompt
                }
            ],
            max_tokens=500,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None

def repair_json(json_str):
    """Attempt to repair common JSON syntax errors"""
    
    try:
        # First attempt - try parsing as is
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Common repair attempts
    repairs = [
        # Fix missing commas
        lambda s: re.sub(r'}\s*{', '},{', s),
        lambda s: re.sub(r']\s*{', '],{', s),
        lambda s: re.sub(r'}\s*\[', '},[', s),
        
        # Fix trailing commas
        lambda s: re.sub(r',\s*}', '}', s),
        lambda s: re.sub(r',\s*]', ']', s),
        
        # Fix missing quotes around keys
        lambda s: re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', s),
        
        # Fix unescaped quotes
        lambda s: s.replace('\\"', '"').replace('"', '\\"'),
        lambda s: re.sub(r'\\*"', '"', s),
        
        # Fix incomplete objects/arrays
        lambda s: s + '}' if s.count('{') > s.count('}') else s,
        lambda s: s + ']' if s.count('[') > s.count(']') else s,
    ]
    
    for repair in repairs:
        try:
            repaired = repair(json_str)
            return json.loads(repaired)
        except:
            continue
    
    # If all repairs fail, try to extract a minimal valid structure
    try:
        # Look for any valid ingredient-like patterns
        ingredient_pattern = r'"name"\s*:\s*"([^"]*)".*?"halal_status"\s*:\s*"([^"]*)".*?"reason"\s*:\s*"([^"]*)"'
        ingredients = []
        
        for match in re.finditer(ingredient_pattern, json_str, re.DOTALL):
            name, status, reason = match.groups()
            ingredients.append({
                "name": name,
                "original_text": name,
                "halal_status": status if status in ['halal', 'haram', 'syubhah'] else 'syubhah',
                "reason": reason
            })
        
        # Extract overall status and confidence if possible
        overall_status = 'syubhah'
        confidence = 30
        concerns = 'JSON parsing issues encountered'
        
        status_match = re.search(r'"overall_halal_status"\s*:\s*"([^"]*)"', json_str)
        if status_match:
            status = status_match.group(1)
            if status in ['halal', 'haram', 'syubhah']:
                overall_status = status
        
        conf_match = re.search(r'"overall_halal_confidence"\s*:\s*(\d+)', json_str)
        if conf_match:
            confidence = int(conf_match.group(1))
        
        concern_match = re.search(r'"main_concerns"\s*:\s*"([^"]*)"', json_str)
        if concern_match:
            concerns = concern_match.group(1)
        
        return {
            "ingredients": ingredients if ingredients else [{
                "name": "Parsing incomplete",
                "original_text": "JSON structure damaged",
                "halal_status": "syubhah", 
                "reason": "Could not parse AI response properly"
            }],
            "overall_halal_status": overall_status,
            "overall_halal_confidence": confidence,
            "main_concerns": concerns
        }
        
    except Exception:
        return None

def parse_halal_response(response_text):
    """Parse and validate the halal analysis response with advanced error handling"""
    
    if not response_text or not response_text.strip():
        return create_default_response("Empty AI response")
    
    try:
        response_text = response_text.strip()
        
        # Check if response is too short
        if len(response_text) < 20:
            return create_default_response("Response too short")
        
        # Clean up the response
        json_str = response_text
        
        # Remove markdown code blocks
        if "```json" in json_str:
            start = json_str.find("```json") + 7
            end = json_str.find("```", start)
            json_str = json_str[start:end] if end != -1 else json_str[start:]
        elif "```" in json_str:
            start = json_str.find("```") + 3
            end = json_str.find("```", start)
            json_str = json_str[start:end] if end != -1 else json_str[start:]
        
        # Remove any text before first { and after last }
        if "{" in json_str and "}" in json_str:
            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}") + 1
            json_str = json_str[start_idx:end_idx]
        else:
            return create_default_response("No valid JSON structure found")
        
        # Basic cleanup
        json_str = json_str.replace("'", '"')
        json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        
        # Attempt to repair and parse JSON
        data = repair_json(json_str)
        
        if data is None:
            return create_default_response("JSON repair failed - structure too damaged")
        
        # Validate and return
        return validate_halal_data(data)
        
    except Exception as e:
        return create_default_response(f"Parsing exception: {str(e)[:100]}")


def create_minimal_analysis_from_text(extracted_text):
    """Create a basic analysis when AI parsing completely fails"""
    
    # Simple keyword-based analysis as absolute fallback
    text_lower = extracted_text.lower()
    
    # Check for obvious haram ingredients
    haram_keywords = ['pork', 'bacon', 'ham', 'wine', 'alcohol', 'gelatin', 'lard']
    syubhah_keywords = ['emulsifier', 'lecithin', 'mono', 'diglyceride', 'enzyme', 'glycerol']
    
    found_haram = [kw for kw in haram_keywords if kw in text_lower]
    found_syubhah = [kw for kw in syubhah_keywords if kw in text_lower]
    
    ingredients = []
    
    # Add found problematic ingredients
    for ingredient in found_haram:
        ingredients.append({
            "name": ingredient.title(),
            "original_text": f"Found in text: {ingredient}",
            "halal_status": "haram", 
            "reason": f"Contains prohibited ingredient: {ingredient}"
        })
    
    for ingredient in found_syubhah:
        ingredients.append({
            "name": ingredient.title(),
            "original_text": f"Found in text: {ingredient}",
            "halal_status": "syubhah",
            "reason": f"Questionable ingredient requiring verification: {ingredient}"
        })
    
    # If no specific ingredients found, create generic response
    if not ingredients:
        ingredients = [{
            "name": "Could not parse ingredients",
            "original_text": "Text analysis incomplete",
            "halal_status": "syubhah",
            "reason": "Unable to properly analyze ingredients. Manual review recommended."
        }]
    
    # Determine overall status
    if found_haram:
        overall_status = "haram"
        confidence = 70
        concerns = f"Contains prohibited ingredients: {', '.join(found_haram)}"
    elif found_syubhah:
        overall_status = "syubhah"
        confidence = 50
        concerns = f"Contains questionable ingredients: {', '.join(found_syubhah)}"
    else:
        overall_status = "syubhah"
        confidence = 30
        concerns = "Could not properly analyze ingredients from the image text"
    
    return {
        "ingredients": ingredients,
        "overall_halal_status": overall_status,
        "overall_halal_confidence": confidence,
        "main_concerns": concerns
    }

def create_default_response(reason):
    """Create a default response when parsing fails"""
    return {
        "ingredients": [
            {
                "name": "Analysis incomplete",
                "original_text": "Could not parse AI response",
                "halal_status": "syubhah",
                "reason": f"Analysis failed: {reason}"
            }
        ],
        "overall_halal_status": "syubhah",
        "overall_halal_confidence": 30,
        "main_concerns": f"Analysis failed: {reason}"
    }

def validate_halal_data(data):
    """Validate and fix parsed JSON data"""
    # Ensure all required fields exist
    if 'overall_halal_status' not in data:
        data['overall_halal_status'] = 'syubhah'
    
    if 'overall_halal_confidence' not in data:
        data['overall_halal_confidence'] = 50
    
    if 'main_concerns' not in data:
        data['main_concerns'] = "Incomplete analysis"
    
    # Ensure ingredients array exists
    if 'ingredients' not in data:
        data['ingredients'] = []
    
    # If ingredients is empty or not a list, create a default entry
    if not isinstance(data['ingredients'], list) or len(data['ingredients']) == 0:
        data['ingredients'] = [
            {
                "name": "No ingredients detected",
                "original_text": "Analysis incomplete",
                "halal_status": "syubhah",
                "reason": "Could not identify ingredients"
            }
        ]
    
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
        
        # Ensure original_text exists
        if 'original_text' not in ingredient:
            ingredient['original_text'] = ingredient.get('name', 'Unknown')
    
    # Validate halal status
    valid_statuses = ['halal', 'haram', 'syubhah']
    if data['overall_halal_status'].lower() not in valid_statuses:
        data['overall_halal_status'] = 'syubhah'
    
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

def hybrid_analysis(client, uploaded_file):
    """Perform simple text-based analysis without JSON complexity"""
    
    # Extract text
    extracted_text = extract_text_with_gpt4o_mini(client, uploaded_file)
    
    if not extracted_text or len(extracted_text.strip()) < 10:
        return None
    
    # Get simple analysis
    response = analyze_halal_status_simple(client, extracted_text)
    
    if response and len(response.strip()) > 20:
        return response
    
    return None

def display_simple_results(analysis):
    """Display results from simple text-based analysis"""
    
    status = analysis['status']
    confidence = analysis['confidence']
    concerns = analysis['concerns']
    problematic_ingredients = analysis['problematic_ingredients']
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Large, prominent status display
        if status == 'halal':
            st.success("### ✅ HALAL")
            st.success(f"**Confidence: {confidence}%**")
            if confidence >= 90:
                st.balloons()
            st.markdown("This product appears to comply with JAKIM Malaysia halal standards.")
        
        elif status == 'haram':
            st.error("### ❌ HARAM")
            st.error(f"**Confidence: {confidence}%**")
            st.markdown("This product contains ingredients that are prohibited under JAKIM halal standards.")
        
        else:  # syubhah
            st.warning("### ⚠️ SYUBHAH (Doubtful)")
            st.warning(f"**Confidence: {confidence}%**")
            st.markdown("This product contains questionable ingredients under JAKIM halal standards. Exercise caution.")
    
    # Display main concerns
    if concerns and concerns.strip() and concerns != "No specific concerns identified":
        st.info(f"**Key Concerns:** {concerns}")
    
    st.markdown("---")
    
    # Display problematic ingredients if any
    if problematic_ingredients and len(problematic_ingredients) > 0:
        # Check if there are actual problematic ingredients vs just default messages
        has_real_ingredients = any(
            ingredient for ingredient in problematic_ingredients 
            if ingredient not in ["See main concerns section for details", "Could not analyze ingredients properly"]
        )
        
        if has_real_ingredients:
            st.header("⚠️ Ingredients of Concern")
            st.markdown("*Showing problematic ingredients that require attention*")
            
            for i, ingredient in enumerate(problematic_ingredients, 1):
                if ingredient in ["See main concerns section for details", "Could not analyze ingredients properly"]:
                    continue
                
                # Parse ingredient info
                if ":" in ingredient and " - " in ingredient:
                    # Format: "Name: Status - Reason"
                    name_status, reason = ingredient.split(" - ", 1)
                    if ":" in name_status:
                        name, status_text = name_status.split(":", 1)
                        name = name.strip()
                        status_text = status_text.strip().upper()
                        reason = reason.strip()
                    else:
                        name = name_status.strip()
                        status_text = "SYUBHAH"
                        reason = ingredient
                else:
                    name = ingredient
                    status_text = "SYUBHAH"
                    reason = "Requires further investigation"
                
                # Display with appropriate styling
                if "HARAM" in status_text:
                    st.error(f"❌ **{name}** - HARAM")
                    st.error(f"**Reason:** {reason}")
                else:
                    st.warning(f"⚠️ **{name}** - SYUBHAH")
                    st.warning(f"**Reason:** {reason}")
                
                st.markdown("---")
        
        else:
            # No specific problematic ingredients identified
            if status in ['haram', 'syubhah']:
                st.header("⚠️ Analysis Summary")
                st.markdown("*Concerns identified but specific ingredients may require manual review*")
                
                for ingredient in problematic_ingredients:
                    if status == 'haram':
                        st.error(f"❌ {ingredient}")
                    else:
                        st.warning(f"⚠️ {ingredient}")
    
    else:
        # No problematic ingredients found
        if status == 'halal':
            st.success("### ✅ No Ingredients of Concern Found")
            st.markdown("All identified ingredients appear to be **Halal** according to JAKIM standards.")
        else:
            st.info("### 📋 General Assessment")
            st.markdown("Detailed ingredient breakdown not available. Please review main concerns above.")
    
    # Show raw analysis for transparency
    if 'full_response' in analysis:
        with st.expander("📋 Full Analysis Details", expanded=False):
            st.text(analysis['full_response'])
    
    # Explanation of JAKIM standards
    with st.expander("ℹ️ About JAKIM MS 1500:2019 Standards"):
        st.markdown("""
        **JAKIM (Department of Islamic Development Malaysia)** sets the world's most stringent halal certification standards based on MS 1500:2019:
        
        - **Zero tolerance** for alcohol from khamr (fermented/distilled sources)
        - **Strict verification** of animal sources and slaughter methods per Hukum Syarak
        - **Comprehensive assessment** of all E-numbers and processing aids per MS 1500:2019
        - **Conservative approach** to questionable ingredients (marked as Syubhah)
        - **Cross-contamination prevention** throughout the entire supply chain
        
        **Status Meanings:**
        - **Halal**: Permissible according to Islamic law and MS 1500:2019 standards
        - **Haram**: Prohibited according to Islamic law and JAKIM classification
        - **Syubhah**: Doubtful/questionable - best to avoid when possible per conservative JAKIM approach
        """).columns(3)
        with col_summary1:
            st.metric("❌ Haram Ingredients", haram_count)
        with col_summary2:
            st.metric("⚠️ Syubhah Ingredients", syubhah_count)
        with col_summary3:
            total_problematic = haram_count + syubhah_count
            st.metric("🔍 Total Concerns", total_problematic)
        
        # Recommendation based on findings
        if haram_count > 0:
            st.error("**JAKIM Recommendation:** ❌ **AVOID** - Contains prohibited (Haram) ingredients according to MS 1500:2019 standards.")
        elif syubhah_count > 0:
            st.warning("**JAKIM Recommendation:** ⚠️ **EXERCISE CAUTION** - Contains doubtful (Syubhah) ingredients. Consider alternatives when possible.")
    
    else:
        # No problematic ingredients found
        st.success("### ✅ No Ingredients of Concern Found")
        st.markdown("All identified ingredients appear to be **Halal** according to JAKIM MS 1500:2019 standards.")
        
        # Still show total count for transparency
        total_ingredients = len(analysis['ingredients'])
        st.info(f"**Total ingredients analyzed:** {total_ingredients}")
    
    # Explanation of JAKIM standards
    with st.expander("ℹ️ About JAKIM MS 1500:2019 Standards"):
        st.markdown("""
        **JAKIM (Department of Islamic Development Malaysia)** sets the world's most stringent halal certification standards based on MS 1500:2019:
        
        - **Zero tolerance** for alcohol from khamr (fermented/distilled sources)
        - **Strict verification** of animal sources and slaughter methods per Hukum Syarak
        - **Comprehensive assessment** of all E-numbers and processing aids per MS 1500:2019
        - **Conservative approach** to questionable ingredients (marked as Syubhah)
        - **Cross-contamination prevention** throughout the entire supply chain
        
        **Status Meanings:**
        - **Halal**: Permissible according to Islamic law and MS 1500:2019 standards
        - **Haram**: Prohibited according to Islamic law and JAKIM classification
        - **Syubhah**: Doubtful/questionable - best to avoid when possible per conservative JAKIM approach
        """)

def get_jakim_guidance(english_name, original_text, status):
    """Provide specific JAKIM guidance based on ingredient type"""
    
    ingredient_text = f"{english_name} {original_text}".lower()
    
    # Common problematic ingredients with specific JAKIM guidance
    guidance_map = {
        'gelatin': "JAKIM requires gelatin to be from halal-certified animal sources or fish. Pork gelatin and uncertified animal gelatin are prohibited under MS 1500:2019.",
        'lecithin': "JAKIM accepts soy and sunflower lecithin as halal. Egg lecithin requires halal certification. Source verification is mandatory per MS 1500:2019.",
        'emulsifier': "JAKIM requires all emulsifiers (E470-E483) to have halal certification or be plant-based. Pork-derived emulsifiers are strictly prohibited.",
        'mono': "Mono- and diglycerides must be from halal-certified sources. JAKIM marks as Syubhah when source is unclear per conservative approach.",
        'diglyceride': "Mono- and diglycerides must be from halal-certified sources. JAKIM marks as Syubhah when source is unclear per conservative approach.",
        'glycerol': "JAKIM prohibits glycerol from pork or non-halal meat sources. Plant-based glycerol is acceptable under MS 1500:2019.",
        'glycerin': "JAKIM prohibits glycerin from pork or non-halal meat sources. Plant-based glycerin is acceptable under MS 1500:2019.",
        'enzyme': "JAKIM requires animal enzymes to be from halal-certified sources. Microbial and plant enzymes are preferred under MS 1500:2019.",
        'alcohol': "JAKIM has zero tolerance for alcohol from khamr sources. Only <0.1% from non-fermented sources may be acceptable.",
        'wine': "Any wine-derived ingredients are prohibited under JAKIM standards, except vinegar (due to istihala - chemical conversion).",
        'carmine': "E120 (Carmine) is classified as Haram by JAKIM as it is derived from insects, which are prohibited.",
        'cochineal': "E120 (Cochineal) is classified as Haram by JAKIM as it is derived from insects, which are prohibited.",
        'whey': "JAKIM requires whey to be from halal-certified rennet sources. Non-certified whey is marked as Syubhah.",
        'cysteine': "L-Cysteine (E920) is Haram if from human hair. JAKIM accepts only synthetic or plant-based L-Cysteine.",
        'riboflavin': "E101 (Riboflavin) is Syubhah if from pork liver/kidney, Halal if from 100% plant sources per JAKIM classification."
    }
    
    # E-number specific guidance
    e_number_guidance = {
        'e120': "Classified as Haram by JAKIM - derived from insects (cochineal/carmine).",
        'e422': "Prohibited if from pork or non-halal meat sources per JAKIM MS 1500:2019.",
        'e441': "Gelatin - Haram unless from halal-certified sources or fish per JAKIM standards.",
        'e471': "Requires halal certification or plant-based source verification per JAKIM.",
        'e472': "Requires halal certification or plant-based source verification per JAKIM.",
        'e542': "Prohibited if from pork or non-halal meat sources per JAKIM classification.",
        'e920': "Haram if from human hair, acceptable if synthetic per JAKIM standards."
    }
    
    # Check for E-numbers first
    for e_num, guidance in e_number_guidance.items():
        if e_num in ingredient_text:
            return guidance
    
    # Check for ingredient keywords
    for keyword, guidance in guidance_map.items():
        if keyword in ingredient_text:
            return guidance
    
    # Default guidance based on status
    if status == 'haram':
        return "This ingredient is prohibited under JAKIM MS 1500:2019 standards. Avoid products containing this ingredient."
    elif status == 'syubhah':
        return "JAKIM takes a conservative approach - when source is unclear or certification absent, ingredients are marked as doubtful. Consider alternatives when possible."
    
    return None

def main():
    st.set_page_config(
        page_title="JAKIM Halal Checker",
        page_icon="🥘",
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    # Web-optimized CSS
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .stButton > button {
        height: 3rem;
        font-size: 1.1rem;
        border-radius: 0.5rem;
    }
    
    .uploadedFile {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    
    .metric-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🥘 JAKIM Halal Checker")
    st.markdown("**Quick halal verification using Malaysia's JAKIM standards**")
    
    # Current standards information
    with st.expander("📋 **Current Standards & References Used** (Click to view)", expanded=False):
        st.markdown("""
        ### 🔄 **Latest Standards Applied (Updated 2025)**
        
        **Primary Reference:**
        - **MS 1500:2019** - Halal Food General Requirements (Third Revision)
        - **Published:** January 3, 2019 (Most Current Version)
        - **Authority:** JAKIM (Department of Islamic Development Malaysia)
        
        **Key Updates in MS 1500:2019:**
        - ✅ Enhanced E-number classifications and restrictions
        - ✅ Updated cross-contamination prevention requirements
        - ✅ Refined alcohol tolerance rules (≤0.1% from non-khamr sources only)
        - ✅ Stricter processing aid verification requirements
        - ✅ Expanded Hukum Syarak compliance criteria
        
        **Referenced Standards:**
        - MS 1500:2019 Halal Food - General Requirements (Third Revision)
        - JAKIM Halal Certification Procedure Manual 2020
        - Trade Descriptions (Use of Expression "Halal") Order 2011
        - Hukum Syarak and Fatwa compliance requirements
        
        **Global Recognition:**
        - 🏆 Malaysia ranks #1 in Global Islamic Economy Indicator 2023
        - 🌍 JAKIM standards accepted by 85+ certification bodies worldwide
        - 📈 Processing time reduced to 15-30 working days (2025 updates)
        """)
    
    st.info("🇲🇾 **JAKIM MS 1500:2019**: Using Malaysia's latest halal standards - the world's most stringent halal certification guidelines")
    
    # Initialize clients
    client = get_openai_client()
    image_processor = get_image_processor()
    
    # File upload section
    st.header("📸 Upload Food Package Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image of the food package (ingredients list should be visible)",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="For best results, ensure the ingredients list is clearly visible and well-lit"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Check file size
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > 10:
            st.error(f"Image too large ({file_size_mb:.1f}MB). Please use an image smaller than 10MB.")
            return
        
        st.success(f"✅ Image uploaded successfully ({file_size_mb:.1f}MB)")
        
        # Analysis section
        st.header("🔍 Halal Analysis")
        
        if st.button("🔍 Analyze Halal Status", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyzing ingredients for halal compliance..."):
                try:
                    # Process image
                    processed_image = image_processor.process_uploaded_file(uploaded_file)
                    
                    if processed_image is None:
                        st.error("❌ Could not process image. Please try a clearer photo.")
                        return
                    
                    # Perform analysis
                    response = hybrid_analysis(client, uploaded_file)
                    
                    if not response:
                        st.error("❌ Analysis failed. Please try with a clearer image of the ingredients list.")
                        
                        # Try emergency fallback analysis
                        with st.spinner("🔄 Trying basic keyword analysis..."):
                            # Get the extracted text again for emergency analysis
                            extracted_text = extract_text_with_gpt4o_mini(client, uploaded_file)
                            if extracted_text and len(extracted_text.strip()) > 10:
                                emergency_analysis = create_basic_keyword_analysis(extracted_text)
                                st.warning("⚠️ Using basic keyword analysis")
                                st.header("📊 Halal Status Results (Basic)")
                                display_simple_results(emergency_analysis)
                                
                                # Show what text was detected
                                with st.expander("📝 Detected Text from Image"):
                                    st.text(extracted_text)
                            else:
                                st.error("❌ Could not extract readable text from the image. Please ensure:")
                                st.markdown("""
                                - The ingredients list is clearly visible
                                - Good lighting without glare
                                - Text is in focus and readable
                                - Image is not blurry or too small
                                """)
                        return

def create_basic_keyword_analysis(extracted_text):
    """Create basic analysis using simple keyword detection"""
    
    text_lower = extracted_text.lower()
    
    # Common problematic ingredients
    haram_keywords = ['pork', 'bacon', 'ham', 'wine', 'alcohol', 'lard', 'gelatin']
    syubhah_keywords = ['emulsifier', 'lecithin', 'mono', 'diglyceride', 'enzyme', 'glycerol', 'e471', 'e472']
    
    found_haram = [kw for kw in haram_keywords if kw in text_lower]
    found_syubhah = [kw for kw in syubhah_keywords if kw in text_lower]
    
    problematic_ingredients = []
    
    # Add found ingredients
    for ingredient in found_haram:
        problematic_ingredients.append(f"{ingredient.title()}: HARAM - Prohibited ingredient under JAKIM standards")
    
    for ingredient in found_syubhah:
        problematic_ingredients.append(f"{ingredient.title()}: SYUBHAH - Requires source verification per JAKIM MS 1500:2019")
    
    # Determine overall status
    if found_haram:
        status = "haram"
        confidence = 70
        concerns = f"Contains prohibited ingredients: {', '.join(found_haram)}"
    elif found_syubhah:
        status = "syubhah"
        confidence = 50
        concerns = f"Contains questionable ingredients: {', '.join(found_syubhah)}"
    else:
        status = "syubhah"
        confidence = 30
        concerns = "Could not identify specific problematic ingredients, but manual review recommended"
        problematic_ingredients = ["Basic analysis complete - manual verification recommended"]
    
    return {
        "status": status,
        "confidence": confidence,
        "concerns": concerns,
        "problematic_ingredients": problematic_ingredients,
        "full_response": f"Basic keyword analysis of: {extracted_text[:200]}..."
    }
                    
                    # Parse and display results
                    analysis = parse_simple_analysis(response)
                    
                    if analysis:
                        st.header("📊 Halal Status Results")
                        display_simple_results(analysis)
                    else:
                        st.error("❌ Could not analyze ingredients. Please ensure the ingredients list is clearly visible.")
                        
                        # Show debug info for troubleshooting
                        if response:
                            with st.expander("🔧 Debug Information", expanded=False):
                                st.text("AI Response:")
                                st.text(response)
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
    
    else:
        # Instructions
        st.markdown("""
        ### 📝 How to use this tool:
        
        1. **Upload an image** of your food package using the file uploader above
        2. **Ensure the ingredients list is clearly visible** and well-lit
        3. **Click "Analyze Halal Status"** to get the halal compliance assessment
        4. **Review the results** based on JAKIM Malaysia standards
        
        ### 💡 Tips for best results:
        - Use good lighting when taking photos
        - Make sure the ingredients text is clear and readable
        - Avoid glare and shadows on the package
        - Include the entire ingredients list in the image
        """)
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About JAKIM")
        st.markdown("""
        **JAKIM** (Department of Islamic Development Malaysia) is the Malaysian government agency responsible for halal certification.
        
        Their standards are considered among the most stringent globally and are widely accepted internationally.
        
        **Key Principles:**
        - Zero tolerance for prohibited substances
        - Strict verification of sources
        - Conservative approach to doubtful ingredients
        """)
        
        st.header("⚠️ Disclaimer")
        st.markdown("""
        This tool provides preliminary halal assessment based on JAKIM standards. 
        
        For official halal certification, consult JAKIM or other authorized halal certification bodies.
        """)

if __name__ == "__main__":
    main()