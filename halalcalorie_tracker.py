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
    
    analysis_prompt = f"""You are a halal food analyst following JAKIM (Jabatan Kemajuan Islam Malaysia) halal standards. Analyze the following extracted text from a food package for halal compliance.

Extracted text from package:
{extracted_text}

CRITICAL INSTRUCTIONS:
1. Analyze ALL ingredients found in the text
2. Follow JAKIM halal standards strictly
3. Respond with ONLY valid JSON - no explanations, no markdown
4. MUST include ingredients array even if empty

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
  "overall_halal_status": "halal",
  "overall_halal_confidence": 85,
  "main_concerns": "brief summary of main issues per JAKIM standards"
}}

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

def parse_halal_response(response_text):
    """Parse and validate the halal analysis response"""
    
    if not response_text or not response_text.strip():
        return create_default_response("Empty AI response")
    
    try:
        response_text = response_text.strip()
        
        # Extract JSON
        json_str = response_text
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            if json_end != -1:
                json_str = response_text[json_start:json_end].strip()
            else:
                json_str = response_text[json_start:].strip()
        
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
            return create_default_response("No JSON found")
        
        # Common JSON fixes
        json_str = json_str.replace("'", '"')
        json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        
        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return create_default_response("JSON parsing failed")
        
        return validate_halal_data(data)
        
    except Exception:
        return create_default_response("Parsing error")

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
    """Perform hybrid analysis with OCR and halal assessment"""
    
    # Extract text
    extracted_text = extract_text_with_gpt4o_mini(client, uploaded_file)
    
    if not extracted_text or len(extracted_text.strip()) < 10:
        return None
    
    # Analyze halal status
    analysis_response = analyze_halal_status(client, extracted_text)
    
    if analysis_response:
        response_clean = analysis_response.strip()
        if response_clean and response_clean.startswith('{') and len(response_clean) > 30:
            return analysis_response
    
    return None

def display_halal_results(analysis):
    """Display streamlined halal status results with focus on problematic ingredients"""
    
    confidence = analysis['overall_halal_confidence']
    status = analysis['overall_halal_status'].lower()
    
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
    
    # Display main concerns if any
    if analysis['main_concerns'] and analysis['main_concerns'].strip():
        st.info(f"**Key Concerns:** {analysis['main_concerns']}")
    
    st.markdown("---")
    
    # Filter and display ONLY problematic ingredients (Haram and Syubhah)
    problematic_ingredients = [
        ingredient for ingredient in analysis['ingredients']
        if ingredient['halal_status'].lower() in ['haram', 'syubhah']
    ]
    
    if problematic_ingredients:
        st.header("⚠️ Ingredients of Concern")
        st.markdown("*Only showing Haram and Syubhah ingredients that require attention*")
        
        # Sort by severity: Haram first, then Syubhah
        def get_severity(ingredient):
            return 1 if ingredient['halal_status'].lower() == 'haram' else 2
        
        sorted_problematic = sorted(problematic_ingredients, key=get_severity)
        
        for i, ingredient in enumerate(sorted_problematic, 1):
            status_lower = ingredient['halal_status'].lower()
            
            # Status styling and icons
            if status_lower == "haram":
                icon = "❌"
                status_text = "HARAM"
                status_color = "red"
                alert_type = "error"
            else:  # syubhah
                icon = "⚠️"
                status_text = "SYUBHAH"
                status_color = "orange"
                alert_type = "warning"
            
            # Get ingredient names
            english_name = ingredient.get('name', 'Unknown ingredient')
            original_text = ingredient.get('original_text', '')
            reason = ingredient.get('reason', 'No explanation provided')
            
            # Create expandable section for each problematic ingredient
            with st.expander(f"{icon} **{status_text}** - {english_name}", expanded=True):
                col_left, col_right = st.columns([2, 3])
                
                with col_left:
                    st.markdown(f"**English Name:** {english_name}")
                    if original_text and original_text != english_name:
                        st.markdown(f"**Original Text:** {original_text}")
                    st.markdown(f"**Status:** {ingredient['halal_status'].title()}")
                
                with col_right:
                    st.markdown("**JAKIM MS 1500:2019 Explanation:**")
                    if alert_type == "error":
                        st.error(f"🚫 **Why HARAM:** {reason}")
                    else:
                        st.warning(f"⚠️ **Why SYUBHAH:** {reason}")
                
                # Add specific JAKIM guidance based on ingredient type
                guidance = get_jakim_guidance(english_name, original_text, status_lower)
                if guidance:
                    st.info(f"**JAKIM Guidance:** {guidance}")
        
        # Summary of problematic ingredients
        haram_count = sum(1 for ing in problematic_ingredients if ing['halal_status'].lower() == 'haram')
        syubhah_count = sum(1 for ing in problematic_ingredients if ing['halal_status'].lower() == 'syubhah')
        
        st.markdown("---")
        st.markdown("### 📊 Summary of Concerns")
        
        col_summary1, col_summary2, col_summary3 = st.columns(3)
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
        page_title="Halal Checker",
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
    st.title("🥘 Halal Checker")
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
                        return
                    
                    # Parse and display results
                    analysis = parse_halal_response(response)
                    
                    if analysis and 'ingredients' in analysis:
                        st.header("📊 Halal Status Results")
                        display_halal_results(analysis)
                    else:
                        st.error("❌ Could not analyze ingredients. Please ensure the ingredients list is clearly visible.")
                        
                        # Show debug info for troubleshooting
                        if response:
                            with st.expander("🔧 Debug Information", expanded=False):
                                st.text("Raw AI Response:")
                                st.code(response[:500] + "..." if len(response) > 500 else response)
                                if analysis:
                                    st.text("Parsed Analysis:")
                                    st.json(analysis)
                
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