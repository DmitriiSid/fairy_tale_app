import random
import streamlit as st 
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from dotenv import load_dotenv
from PIL import Image
import os
from io import BytesIO
import openai
from openai import OpenAI
import base64
import requests

import nltk
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

if 'punkt_downloaded' not in st.session_state:
    nltk.download('punkt')  
    st.session_state['punkt_downloaded'] = True
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_file= load_lottie("https://lottie.host/b0116ab0-d32f-4cc2-9c0b-d6e5be1022dc/xegcOEi7s8.json")
lottie_pig = load_lottie("https://lottie.host/8fc522c9-2e53-4bda-a9c2-dda22dea81c5/ly1ZgXFc25.json")

st.set_page_config(page_title="Fairy Tale App",
        page_icon="🔮",
        layout="wide",
        )

emotional_or_action_words = {
    'adventure', 'amazed', 'angry', 'beautiful', 'benevolent', 'bewitched', 'blessed', 'brave',
    'bright', 'calm', 'caring', 'cheerful', 'courage', 'courageous', 'cruel', 'cry', 'cunning',
    'cursed', 'danger', 'daring', 'dark', 'deceptive', 'defeated', 'despairing', 'discover',
    'dream', 'enchanted', 'escape', 'evil', 'excited', 'fateful', 'fear', 'fearless', 'foolish',
    'forgiving', 'fortunate', 'found', 'friend', 'frightened', 'fun', 'funny', 'generous', 'gentle',
    'greedy', 'grim', 'happy', 'hateful', 'help', 'hero', 'heroic', 'hidden', 'hide', 'honest',
    'hopeful', 'joy', 'joyful', 'kindhearted', 'laugh', 'lonely', 'lost', 'love', 'loving', 'loyal',
    'magic', 'magical', 'mysterious', 'mystery', 'noble', 'noisy', 'old', 'peaceful', 'play', 'poor',
    'powerful', 'protect', 'rescue', 'revealed', 'rich', 'sad', 'scared', 'search', 'silent', 'sneaky',
    'sorrowful', 'surprise', 'treacherous', 'treasure', 'trick', 'triumphant', 'ugly', 'vengeful',
    'vicious', 'warlike', 'weak', 'whimsical', 'wicked', 'wise', 'wish', 'wonder', 'young'
}

def about_page():
    st.write("This app was developed as a term project")

def build_matcher():
    matcher = Matcher(nlp.vocab)
    
    # Patterns for fairy tale elements
    patterns = [
        [{"POS": "NOUN"}, {"POS": "VERB"}],  # Simple noun-verb pairs
        [{"LEMMA": {"IN": ["find", "discover", "rescue", "escape", "reveal"]}}],  # Action verbs
        [{"LEMMA": {"IN": ["magic", "princess", "castle", "dragon", "witch", "fairy"]}}],  # Fairy tale nouns
        [{"POS": "ADJ"}, {"POS": "NOUN"}]  # Adjective-noun pairs (e.g., 'golden hair')
    ]
    for pattern in patterns:
        matcher.add("FAIRY_TALE_PATTERN", [pattern])
    return matcher

def clean_scene(scene): 
    scene = scene.replace("\n", " ").strip() 
    scene = scene.replace('"', " ").strip() 
    return scene

def extract_key_scenes(text, top_n=5):
    doc = nlp(text)
    matcher = build_matcher()
    matches = matcher(doc)

    # Extract sentences that contain matched patterns
    key_sentences = set()
    for match_id, start, end in matches:
        span = doc[start:end].sent  # Get the whole sentence that contains the match
        cleaned_scene = clean_scene(span.text)
        key_sentences.add(cleaned_scene)

    # Sort and select top_n sentences
    return sorted(list(key_sentences))[:top_n]

def create_descriptive_prompt(scenes):

    descriptive_adjectives = ['enchanting', 'mysterious', 'serene', 'joyful', 'dramatic', 'delicate', 'delicious', 'delighted', 'delightful', 'delinquent', 'delirious', 'deliverable', 'deluded', 'demanding', 'demented', 'democratic', 'demonic', 'demonstrative', 'demure', 'deniable', 'dense', 'dependable', 'dependent', 'deplorable', 'deploring', 'depraved', 'depressed', 'depressing', 'depressive', 'deprived', 'deranged', 'derivative', 'derogative', 'derogatory', 'descriptive', 'deserted', 'designer', 'desirable', 'desirous', 'desolate', 'despairing', 'desperate', 'despicable']
    sensory_phrases = [
    'under the twinkling stars', 'in the lush, green garden', 'by the sparkling river','amidst the whispering woods','over the rolling, fog-covered hills','beside the crackling campfire','within the ancient, echoing halls','underneath the bright, full moon','along the bustling, cobblestone streets','near the fragrant, blooming meadows','atop the windswept, rugged cliffs','inside the warm, cozy cottage','among the vibrant, bustling market','beneath the serene, azure sky','through the dense, misty rainforest','across the vast, sunbaked desert','within the quiet, hallowed chapel','along the gentle, babbling brook','in the midst of the thunderous storm','under the canopy of twinkling stars'
    ]
    emotional_phrases = [
    'with a sense of wonder','feeling brave and bold','with a heart full of hope',
    'overwhelmed with joy and excitement','shrouded in a veil of mystery','carrying the weight of sadness',
    'filled with a burning curiosity','lost in a sea of confusion and doubt','radiating with unbridled happiness','consumed by a deep, unyielding anger','embracing a moment of peaceful serenity',
    'overcome with a wave of nostalgia','bursting with pride and accomplishment','engulfed in the depths of despair','floating in a dream of whimsical fantasy','driven by a relentless thirst for adventure',
    'soaring on the wings of freedom','trembling with fear and apprehension',
    'awash in a flood of relief','drowning in a pool of regret and remorse']

    prompts = []
    for scene in scenes:
        # Randomly choose descriptors to add richness
        adjective = random.choice(descriptive_adjectives)
        sensory_detail = random.choice(sensory_phrases)
        emotional_detail = random.choice(emotional_phrases)

        # Create and return the prompt
        prompt = f"{adjective} scene of {scene} {sensory_detail}, {emotional_detail}."
        prompts.append(prompt)
    
    return prompts


HUGGING_FACE_API_TOKEN = st.secrets["HUGGING_FACE_API_TOKEN"]
#headers = {"Authorization": f"""Bearer {st.secrets["HUGGING_FACE_API_TOKEN"]}"""}

def query(payload):
    headers = {"Authorization": f"""Bearer {HUGGING_FACE_API_TOKEN}"""}
    API_URL = "https://api-inference.huggingface.co/models/prompthero/openjourney"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def generate_images(prompts): # HuggingFace model
    images =[]
    for prompt in prompts:
        image_bytes = query({ "inputs": f"{prompt}",})
        images.append(image_bytes)
    
    return images

def generate_images_using_openai(model,prompts):
    images = []
    output_directory = "./images/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for i, prompt in enumerate(prompts):
        # Generate the image response
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
            quality="hd",
            n=1,
        )
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        img = Image.open(BytesIO(image_response.content))
        output_filename = f"image_{i}.png"
        img.save(os.path.join(output_directory, output_filename), 'PNG')

        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        images.append(img_byte_arr)
    return images

def download_and_save_images(images):
    i = 1
    output_directory = "./images/"

    # Create the 'images' directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for image in images:
        try:
            output_filename = f"image_{i}.png"

            # Open the image using PIL
            img = Image.open(BytesIO(image))

            # Save the image as PNG in the 'images' directory
            img.save(os.path.join(output_directory, output_filename), 'PNG')

            print(f"Image saved as {output_filename}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        i += 1

def encode_image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def display_example():
    story = """
    Once upon a time, in the kingdom of Enchantia, there lived a beautiful princess named Fiona. Fiona had long golden hair, sparkling blue eyes, and a heart filled with love for one thing in particular - cats. She adored everything about them, from their soft fur to their playful nature.

    Every day, Fiona would spend her time in the royal gardens, surrounded by a multitude of cats. They would purr and rub against her legs, making her giggle with joy. She would name each cat and spend hours playing and cuddling with them.

    One sunny morning, while Fiona was feeding her feline friends, she discovered a tiny, fluffy kitten hiding in the flower bushes. The poor little creature had been abandoned and was all alone. Fiona's heart ached with sadness for this little kitten.

    Without a second thought, Fiona scooped up the kitten and held her close. She could feel the tiny heartbeat and saw the fear in the kitten's eyes. Determined to provide a loving home, Fiona carried the kitten to the castle, seeking assistance from her parents, the king, and queen.

    When the king and queen saw the tearful princess holding the kitten, their hearts melted. They gave their approval for Fiona to keep the little furball and adopted her as a member of the royal household. Fiona named the kitten Daisy, and from that day on, they became inseparable.

    Daisy and Fiona spent their days exploring the castle corridors, chasing butterflies, and playing hide-and-seek. Daisy grew stronger and healthier under Fiona's loving care. The bond between them was so strong that they could communicate without words.

    One evening, a terrible storm swept through the kingdom. Thunder roared, and lightning streaked across the sky, filling the air with a sense of danger. Fiona, who was afraid of storms herself, found comfort in Daisy's presence. As Fiona clung to her precious feline friend, she whispered, "Don't be afraid, Daisy. I'll protect you, just like you protect me."

    As the storm raged on, the windows shook, and the rain poured heavily. Suddenly, a blinding bolt of lightning struck the castle, causing a fire to ignite in the tower. Panic flooded the kingdom, and people ran in all directions, seeking safety.

    Fiona, with Daisy in her arms, knew she had to do something. Filled with bravery and love, she ran towards the tower, saving those who were trapped. Her heart raced

    """
    key_scenes = ["As Fiona clung to her precious feline friend, she whispered,  Don't be afraid, Daisy.",
    'As the storm raged on, the windows shook, and the rain poured heavily.',
    'Daisy (cat) and Fiona spent their days exploring the castle corridors, chasing butterflies, and playing hide-and-seek.',
    'Determined to provide a loving home, Fiona carried the kitten to the castle, seeking assistance from her parents, the king, and queen.',
    'Fiona had long golden hair, sparkling blue eyes, and a heart filled with love for one thing in particular - cats.']

    images_paths = ['./images/image_1.png',
    './images/image_2.png',
    './images/image_3.png',
    './images/image_4.png',
    './images/image_5.png']

    markdown_text = ""

    # Iterate through key scenes and images
    for scene, image_path in zip(key_scenes, images_paths):
        # Split the text at the key scene
        parts = story.split(scene, 1)
        markdown_text += parts[0]
        
        # Encode the image
        base64_image = encode_image_to_base64(image_path)

        # Insert the image tag
        markdown_text += f'<img src="data:image/png;base64,{base64_image}" style="width: 50%; margin-left: 5px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); border-radius: 0.5px; transition: transform 0.3s ease, box-shadow 0.3s ease;">'

        # Add the scene text back
        markdown_text += scene

        # Update the story_text to the remaining part after the key scene
        story_text = parts[1] if len(parts) > 1 else ''

    # Add any remaining text after the last image
    markdown_text += story_text
    st.markdown(markdown_text, unsafe_allow_html=True)


#Streamlit Code
st.sidebar.info("🤖 Application settings 🤖")
sd1, sd2 = st.sidebar.columns(2)
lm = sd1.selectbox("Select model for text generation", ["Mistral","GPT"])
image_model = sd2.selectbox("Select model for image generation", ["OpenJourney", "DALL-E-2", "DALL-E-3"])
user_key = st.container ()
if lm == "GPT" or image_model == "DALL-E-2" or image_model == "DALL-E-3":
    user_key = st.sidebar.text_input(label = " Enter your OpenAI API key",type='password')
#text2speech = st.sidebar.radio("Text to speech 👇", ["Actve", "Inactve"], horizontal=True)
st.sidebar.info("🏰 Fairy Tale settings 🏰")
#sd1, sd2 = st.sidebar.columns(2)
gender = st.sidebar.radio("Select gender ", ["Boy", "Girl", "Diverse"], horizontal=True)
#child_name = sd2.text_input(label="Enter child's name ",placeholder="Optional")
age = st.sidebar.select_slider(
    "Select kid's age",
    options=[1,2,3,4,5,6,7,8,9,10,11,12],value=5)
#text2speech_yes = sd1.checkbox( "Text to speech - actve")
#text2speech_no = sd2.checkbox( "Text to speech - inactve")
if 'api_key' not in st.session_state or st.session_state['api_key'] != user_key:
    st.session_state['api_key'] = user_key 
    
characters = st.sidebar.text_input(label="Which characters you want to be included ?", placeholder="Shrek, Puff in boots  ... ")
mood = st.sidebar.text_input(
    label="Mood (e.g. inspirational, funny, serious) (optional)",
    placeholder="inspirational",
)

client = OpenAI(api_key=st.session_state['api_key'])
input_prompt = None

def text_query(payload,max_tokens = 3000 ):
    headers = {"Authorization": f"""Bearer {HUGGING_FACE_API_TOKEN}"""}
    API_URL = "https://api-inference.huggingface.co/models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
    payload["parameters"] = {"max_length": max_tokens} 
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def generate_story_mistral(input_prompt,gender,age,characters,mood):
    formatted_prompt = (f"You are a fairy tale teller; "
                        f"You should generate a story based on these instructions: {input_prompt}. "
                        f"The story should be no more than 500 words; "
                        f"Child's gender is {gender}, child's age is {age}. "
                        f"Make sure to use these characters {characters}. "
                        f"The mood of the story should be {mood}"
                        f"Make sure to end the story and that it is not longer than 700 words")
    response = text_query({"inputs": formatted_prompt})
    # if isinstance(response, list) and len(response) > 0:
    #     content = response[0].get("generated_text", "")  # Assuming the story is in the first element of the list
    # else:
    #     content = ""
    return response

def generate_story(input_prompt,gender,age,characters,mood):
    response = client.chat.completions.create(
  #model="gpt-4-1106-preview",
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": f"""You are a fairy tale teller;\nYou should generate a story based on a thses instructions:{input_prompt}.\nThe story should be no more than 500 words;\n Child's gender is {gender}, childs's age is {age}.\n 
      Make sure to use these characters {characters}. The mood of the story should be {mood}"""
    }
  ],
  temperature=1,
  max_tokens=1024,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
    content  = response.choices[0].message.content
    return content

def continue_story(initial_story, continuation_tokens=100):
    # Take the last part of the initial story as a new prompt
    new_prompt = initial_story[-250:]  # Adjust the character count as needed
    continuation_response = text_query({"inputs": new_prompt}, max_tokens=continuation_tokens)

    if isinstance(continuation_response, list) and len(continuation_response) > 0:
        continuation_content = continuation_response[0].get("generated_text", "")
    else:
        continuation_content = ""

    return initial_story + continuation_content 

app_mode =option_menu(
        menu_title=None,
        options=['Main screen','About this app'],
        icons=['bi-house-fill','bi-info-square-fill'],
        orientation="horizontal"
    )
if app_mode == 'Main screen':
    if 'story' not in st.session_state:
        st.session_state['story'] = ""
    
    st.subheader("This is a Fairy Tale Generation App that uses AI to generates text and images from text prompt.")
    input_prompt = st.text_area(label="Enter a fairy tale description for generation", placeholder="A fairy tale about princess Freya ... ")

    if 'story_generated' not in st.session_state:
        st.session_state.story_generated = False

    col1, col2 = st.columns(2)
    generate_button = col1.button("Generate Fairy Tale")
    display_button = col2.button("Display Generated Story")

    if input_prompt is not None and user_key != '' and generate_button:   
        #if st.button("Generate Fairy Tale"):
        if gender and input_prompt and age and mood and characters:
            with st.spinner('The fairy tale is generating ...'):
                lottie_placeholder = st.empty()
                loading_image = random.uniform(0, 1)
                with lottie_placeholder:
                    if loading_image<=0.5:                            
                        st_lottie(lottie_file)
                    else:
                        st_lottie(lottie_pig)
                if lm == "GPT":
                    story = generate_story(input_prompt,gender,age,characters,mood)
                if lm == "Mistral":
                    story = generate_story_mistral(input_prompt,gender,age,characters,mood)  
                    story = story[0].get("generated_text","")
                    start_phrase = "\n\n"                        
                    if start_phrase in story:
                        content = story.split(start_phrase, 1)[1]  # Keep the part after the phrase
                        story = start_phrase + content                                               
                key_scenes = extract_key_scenes(story)
                prompts = create_descriptive_prompt(key_scenes)
                if image_model == "Huggingface":
                    images = generate_images(prompts)
                    download_and_save_images(images)
                if image_model == "DALL-E-2":
                    images = generate_images_using_openai(model = "dall-e-2",prompts = prompts)
                if image_model == "DALL-E-3": 
                    images = generate_images_using_openai(model = "dall-e-3",prompts = prompts)

                image_directory = "./images/"

                # Initialize an empty list to store image paths
                images_paths = []

                # Check if the directory exists
                if os.path.exists(image_directory) and os.path.isdir(image_directory):
                    # List all files in the directory
                    image_files = os.listdir(image_directory)

                    # Filter only image files (you can customize this to include specific file extensions)
                    image_files = [file for file in image_files if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))]

                    # Create full paths for the image files and add them to the 'images' list
                    images_paths = [os.path.join(image_directory, file) for file in image_files]
                else:
                    print(f"The directory '{image_directory}' does not exist.")

                markdown_text = ""

                # Iterate through key scenes and images
                for scene, image_path in zip(key_scenes, images_paths):
                    # Split the text at the key scene
                    parts = story.split(scene, 1)
                    markdown_text += parts[0]
                    
                    # Encode the image
                    base64_image = encode_image_to_base64(image_path)

                    # Insert the image tag
                    markdown_text += f'<img src="data:image/png;base64,{base64_image}" style="width: 50%; margin-left: 5px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); border-radius: 0.5px; transition: transform 0.3s ease, box-shadow 0.3s ease;">'

                    # Add the scene text back
                    markdown_text += scene

                    # Update the story_text to the remaining part after the key scene
                    story_text = parts[1] if len(parts) > 1 else ''

                # Add any remaining text after the last image
                markdown_text += story_text
            st.session_state['story'] = markdown_text
            lottie_placeholder.empty()  
            st.session_state.story_generated = True
            st.success("The story was generated ✅")
            st.markdown(st.session_state['story'], unsafe_allow_html=True)
            
            #message = st.chat_message("assistant")
            #message.write(generate_story(input_prompt,gender,age,characters,mood ))
            #display_example()
            st.session_state.story_generated = True
            #st.balloons()
        else:
            st.error("Some data is missing, check input options")
    elif display_button:
        if st.session_state.story_generated:
            st.markdown(st.session_state['story'], unsafe_allow_html=True)
        else:
            st.warning("No story has been generated yet. Please generate a story first.")
    if not input_prompt or not user_key:
        col1.warning("Please enter a prompt and set up your API key.")

    # else:
    #     col1, col2 = st.columns(2)
    #     col1.button("Generate Fairy Tale",disabled = True)
    #     col1.warning("Please set up your API key")
    if st.session_state.story_generated:
        if st.button("Delete the story",key="deleteButton"):
            st.session_state.story_generated = False
            st.session_state['story'] = ""
            st.write("The story has been deleted. ✅")

elif app_mode == "About this app":
    about_page()
