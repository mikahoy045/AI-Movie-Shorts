import os
import re
import ast
import requests
from pathlib import Path
import json
import time

def load_config():
    try:
        with open('config.json', 'r') as file:
            config = json.loads(file.read())
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def get_srt_response(script, model="deepseek-r1-distill-llama-70b"):
    config = load_config()
    if not config or 'grooq_api_key' not in config:
        print("Error: Missing or invalid config.json file")
        return None
    
    API_TOKEN = config.get('grooq_api_key')
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    def query(messages, max_retries=3, retry_delay=5):
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": model,
                    "messages": messages
                }
                
                print("\nSending request to Groq API...")
                print(f"Payload: {payload}")
                
                response = requests.post(API_URL, headers=headers, json=payload)
                print(f"\nRaw API Response Status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"API Error: Status code {response.status_code}")
                    print(f"Response: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
                    
                json_response = response.json()
                print(f"\nFull JSON Response: {json_response}")
                
                if json_response and 'choices' in json_response:
                    content = json_response['choices'][0]['message']['content']
                    print(f"\nRaw content: {content}")
                    
                    # Remove the <think> section if present
                    if "<think>" in content and "</think>" in content:
                        content = content.split("</think>")[1].strip()
                        print(f"\nCleaned content (after removing <think>): {content}")
                    return content
                return None
            except Exception as e:
                print(f"Error calling Groq API: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return None
        return None

    # Split the script into smaller chunks
    entries = script.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    max_chunk_length = 4000  # Groq has larger context window
    
    for entry in entries:
        entry_length = len(entry)
        if current_length + entry_length > max_chunk_length:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [entry]
            current_length = entry_length
        else:
            current_chunk.append(entry)
            current_length += entry_length
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    # Process only the first chunk
    if chunks:
        try:
            messages = [
                {"role": "system", "content": "You are a movie expert analyzing scripts."},
                {"role": "user", "content": f"""Process this movie script.
                Extract key scenes with their timestamps and provide brief summaries.
                Format: {{"timestamp": "summary"}}
                Keep summaries concise but informative.
                Content:\n{chunks[0]}"""}
            ]
            
            print("\nProcessing first chunk...")
            print(f"Chunk content: {chunks[0][:200]}...") # Print first 200 chars of chunk
            
            chunk_response = query(messages)
            if chunk_response:
                print("\nSuccessfully processed first chunk")
                print(f"Final processed response: {chunk_response}")
                return chunk_response
            else:
                print("Failed to get valid response for first chunk")
                return None
        
        except Exception as e:
            print(f"Error processing first chunk: {e}")
            return None
    
    print("No chunks to process")
    return None

def read_combined_script(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def main():
    # Create necessary directories
    os.makedirs('scripts/srt_files', exist_ok=True)
    
    movie_title = "Iron_Man"
    num_clips = 25
    combined_script_path = "scripts/srt_files/Iron_Man.srt"
    
    combined_script = read_combined_script(combined_script_path)
    
    if not combined_script:
        print("Error: Could not read SRT file")
        return
    
    # Modify the prompt to work with chunks
    script = f'''Analyze this script of the movie {movie_title} and identify key scenes. 
                Choose time ranges that are most essential to the plot and development of the movie's story. 
                Each chosen range should be between 10-30 seconds.
                Format the output as a Python dictionary: {{"120-145": "PLOT SUMMARY", "280-300": "PLOT SUMMARY"}}
                Each summary should be 3 sentences describing the scene's events.
                The first summary should start with: "Here we go, let's go over the movie {movie_title}."
                Make sure to cover the whole movie's plot arc.'''
    
    response = get_srt_response(script + combined_script)
    
    if response:
        try:
            scene_dict = ast.literal_eval(response)
            
            print("\nProcessed Scene Summaries:")
            print("------------------------")
            for time_range, summary in scene_dict.items():
                print(f"\nTime Range: {time_range}")
                print(f"Summary: {summary}")
                print("------------------------")
                
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response}")
    else:
        print("Failed to get response from Groq")

if __name__ == "__main__":
    main()
