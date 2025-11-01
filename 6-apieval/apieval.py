import os
import csv
import json
from openai import OpenAI
from datetime import datetime
import time
from tqdm import tqdm


client = OpenAI(
    api_key="",  
    base_url="",
)


def get_urls_from_csv(csv_file):
    urls = []
    with open(csv_file, mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  
        for row in csv_reader:
            urls.append(row[0])  
    return urls

def extract_grade_from_url(url):
    try:

        parts = url.split('/')
        
        if len(parts) < 3:
            print(f"warn: {url}")
            return None
        
        grade = int(parts[-2])  
        return grade
    except Exception as e:
        print(f"fail URL: {url}, {str(e)}")
        return None


def update_summary(results, grade_distribution, file_name):
    summary = {
        "evaluated_grades": [0, 1, 2, 3, 4],
        "total_images": len(results),
        "grade_distribution": grade_distribution,
        "results": results
    }
    
    
    output_file = file_name or f"{datetime.now().strftime('%Y-%m-%d')}-summarymodelqwen.json"
    
   
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(summary, json_file, ensure_ascii=False, indent=4)


def batch_evaluate(urls, output_file):
    results = []
    grade_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    for idx, url in tqdm(enumerate(urls), total=len(urls), desc="Processing URLs"):
        grade = extract_grade_from_url(url)
        
        if grade is None:
            print(f"fail: {url}")
            continue  
        
        
        retry_count = 0
        success = False
        while retry_count < 5 and not success:
            try:
                completion = client.chat.completions.create(
                    model="qwen-vl-max-2025-08-13",  # replace
                    messages=[{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": url}},
                            {"type": "text", "text": "You are a professional radiologist. You are provided with a knee X-ray image and you should determine the Kellgren-Lawrence grade of this knee joint X-ray based on the Kellgren-Lawrence grading system.\nThe specific criteria for Kellgren-Lawrence grading system are as follows: Grade 0: No osteoarthritis, No radiographic features of osteoarthritis. Grade 1: Doubtful osteoarthritis, Doubtful narrowing of joint space and possible osteophytic lipping. Grade 2: Mild osteoarthritis, Definite osteophytes and possible narrowing of joint space. Grade 3: Moderate osteoarthritis, Multiple osteophytes, definite narrowing of joint space, some sclerosis, and possible deformity of bone ends. Grade 4: Severe osteoarthritis, Large osteophytes, marked narrowing of joint space, severe sclerosis, and definite deformity of bone ends.\nPlease output the most likely Kellgren-Lawrence grade you determine. The format of your answer should be: The most likely Kellgren-Lawrence grade of this knee X-ray image is Grade {X}: {the description}."}
                        ]}],
                    temperature=0.01,  
                    top_p=0.7,         
                    max_tokens=512     
                )

                
                usage = completion.usage
                print(f"Input tokens: {usage.prompt_tokens}, Output tokens: {usage.completion_tokens}")
                
                predicted_output = completion.model_dump_json()  
                print(f"Predicted output: {predicted_output}")  
                print(f"Predicted output type: {type(predicted_output)}")  
                
                
                if isinstance(predicted_output, str):
                    predicted_output = json.loads(predicted_output)  
                
                
                predicted_text = predicted_output.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                
                image_name = url.split("/")[-1]
                results.append({
                    "true_grade": grade,
                    "image_file": url,
                    "predicted_output": predicted_text,
                    "image_name": image_name
                })
                
                
                if grade in grade_distribution:
                    grade_distribution[grade] += 1
                
                
                update_summary(results, grade_distribution, output_file)
                
                success = True
            except Exception as e:
                retry_count += 1
                print(f"fail: {url}, {str(e)} (retry {retry_count} )")
                time.sleep(10) 
        
        if retry_count == 10 and not success:
            print(f"fail：{url}")
    
    return results, grade_distribution


if __name__ == "__main__":
    csv_file = ""
    urls = get_urls_from_csv(csv_file)
    
    
    output_file = f"{datetime.now().strftime('%Y-%m-%d')}-summarymodelqwenother.json"
    
    
    results, grade_distribution = batch_evaluate(urls, output_file)
    
    
    update_summary(results, grade_distribution, output_file)
    print(f"save {output_file}")

