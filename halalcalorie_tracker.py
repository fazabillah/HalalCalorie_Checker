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

OUTPUT FORMAT - EXACT JSON STRUCTURE:
{{
  "overall_halal_status": "halal",
  "overall_halal_confidence": 85,
  "main_concerns": "brief summary of main issues per JAKIM standards"
}}

JAKIM HALAL STANDARDS - KEY FOCUS AREAS:
1. **Animal-derived ingredients**: Must be from halal animals slaughtered according to Islamic law
2. **Alcohol**: Completely prohibited, including as processing aid or residual amounts
3. **Gelatin**: Only halal if from halal animal sources or fish gelatin
4. **Emulsifiers (E471, E472)**: Haram if from pork, halal if from halal animals/plants
5. **Lecithin**: Halal if from soy/sunflower, haram if from egg (if from non-halal source)
6. **Enzymes**: Must be from halal sources (microbial/plant preferred)
7. **Mono- and Diglycerides**: Haram if from pork fat, halal if from plants/halal animals
8. **Whey**: Halal only if from halal-certified rennet

JAKIM E-NUMBER CLASSIFICATIONS:
- E120 (Carmine): HARAM - from insects
- E441 (Gelatin): SYUBHAH/HARAM unless verified halal source
- E471, E472: SYUBHAH unless certified halal source
- E631, E635: SYUBHAH - may contain pork-derived ingredients
- E920 (L-Cysteine): HARAM if from human hair, halal if synthetic

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
        "overall_halal_status": "syubhah",
        "overall_halal_confidence": 30,
        "main_concerns": f"Analysis failed: {reason}"
    }

def validate_halal_data(data):
    """Validate and fix parsed JSON data"""
    # Ensure required fields exist
    if 'overall_halal_status' not in data:
        data['overall_halal_status'] = 'syubhah'
    
    if 'overall_halal_confidence' not in data:
        data['overall_halal_confidence'] = 50
    
    if 'main_concerns' not in data:
        data['main_concerns'] = "Incomplete analysis"
    
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
    """Display streamlined halal status results"""
    
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
    
    # Explanation of JAKIM standards
    with st.expander("ℹ️ About JAKIM Standards"):
        st.markdown("""
        **JAKIM (Department of Islamic Development Malaysia)** sets some of the world's most stringent halal certification standards:
        
        - **Zero tolerance** for alcohol and pork derivatives
        - **Strict verification** of animal sources and slaughter methods
        - **Comprehensive assessment** of processing aids and additives
        - **Conservative approach** to questionable ingredients (marked as Syubhah)
        
        **Status Meanings:**
        - **Halal**: Permissible according to Islamic law
        - **Haram**: Prohibited according to Islamic law
        - **Syubhah**: Doubtful/questionable - best to avoid when possible
        """)

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
    
    # Info about JAKIM standards
    st.info("🇲🇾 **JAKIM Standards**: Using Malaysia's strict halal certification guidelines - recognized globally for their stringent requirements")
    
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
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
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
                    
                    if analysis:
                        st.header("📊 Halal Status Results")
                        display_halal_results(analysis)
                    else:
                        st.error("❌ Could not analyze ingredients. Please ensure the ingredients list is clearly visible.")
                
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