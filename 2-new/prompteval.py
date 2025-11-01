import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests

from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "LLaVA-Med Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(
                value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown.update(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def add_text(state, text, image, image_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            # text = '<Image><image></Image>' + text
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "llava" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        template_name = "mistral_instruct" # FIXME: overwrite
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


title_markdown = ("""
# 🌋 LLaVA-Med: Large Language and Vision Assistant for Medical Research
[[Project Page]](https://llava-vl.github.io) [[Paper]](https://arxiv.org/abs/2304.08485) [[Code]](https://github.com/haotian-liu/LLaVA) [[Model]](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0)
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

def build_demo(embed_mode):
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="LLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/bio_patch.png", "What is this image about?"],
                    [f"{cur_dir}/examples/med_img_1.png", "Can you describe the image in details?"],   
                    [f"{cur_dir}/examples/xy_chromosome.jpg", "Can you describe the image in details?"],   
                    [f"{cur_dir}/examples/synpic42202.jpg", "Is there evidence of an aortic aneurysm? Please choose from the following two options: [yes, no]?"], # answer" yes 
                    [f"{cur_dir}/examples/synpic32933.jpg", "What is the abnormality by the right hemidiaphragm?"],      # answer: free air                             
                    [f"{cur_dir}/examples/extreme_ironing.jpg", "What is unusual about this image?"],
                    [f"{cur_dir}/examples/waterview.jpg", "What are the things I should be cautious about when I visit here?"],
                ], inputs=[imagebox, textbox])

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chatbot", label="LLaVA-Med Chatbot", height=550)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                _js=get_window_url_params,
                queue=False
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    
    # Batch process
    parser.add_argument("--batch", action="store_true", help="Run in batch mode")
    parser.add_argument("--batch-dir", type=str, default="", 
                       help="Base directory containing grade subdirectories")
    parser.add_argument("--batch-temperature", type=float, default=0.2, 
                       help="Temperature for batch processing")
    
    parser.add_argument("--grades", type=str, default="all", 
                       help="Grades to evaluate: 'all' or comma-separated list (e.g., '0,1,2' or '3' or '0,2,4')")
    
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    
    if args.batch:
        import glob
        from PIL import Image
        
        
        prompt = "You are a professional radiologist. You are provided with a knee X-ray image and you should determine the Kellgren-Lawrence grade of this knee joint X-ray based on the Kellgren-Lawrence grading system.\nThe specific criteria for Kellgren-Lawrence grading system are as follows: Grade 0: No osteoarthritis, No radiographic features of osteoarthritis. Grade 1: Doubtful osteoarthritis, Doubtful narrowing of joint space and possible osteophytic lipping. Grade 2: Mild osteoarthritis, Definite osteophytes and possible narrowing of joint space. Grade 3: Moderate osteoarthritis, Multiple osteophytes, definite narrowing of joint space, some sclerosis, and possible deformity of bone ends. Grade 4: Severe osteoarthritis, Large osteophytes, marked narrowing of joint space, severe sclerosis, and definite deformity of bone ends.\nPlease output the most likely Kellgren-Lawrence grade you determine. The format of your answer should be: The most likely Kellgren-Lawrence grade of this knee X-ray image is Grade {X}: {the description}."
        
        # Eval
        if args.grades.lower() == "all":
            grades_to_evaluate = list(range(5))  # [0, 1, 2, 3, 4]
        else:
            try:
                grades_to_evaluate = [int(g.strip()) for g in args.grades.split(',')]
                
                for g in grades_to_evaluate:
                    if g < 0 or g > 4:
                        logger.error(f"Invalid grade: {g}. Grades must be between 0 and 4.")
                        exit(1)
            except ValueError:
                logger.error(f"Invalid grades format: {args.grades}. Use 'all' or comma-separated numbers (e.g., '0,1,2').")
                exit(1)
        
        logger.info(f"Will evaluate grades: {grades_to_evaluate}")
        
        
        base_dir = args.batch_dir
        all_results = []  
        
        # Check
        for grade in grades_to_evaluate:
            grade_dir = os.path.join(base_dir, str(grade))
            
            
            if not os.path.exists(grade_dir):
                logger.warning(f"Grade {grade} directory not found: {grade_dir}, skipping...")
                continue
                
            logger.info(f"\nProcessing Grade {grade} images from {grade_dir}")
            
            
            image_files = glob.glob(os.path.join(grade_dir, "*.jpg")) + \
                         glob.glob(os.path.join(grade_dir, "*.png")) + \
                         glob.glob(os.path.join(grade_dir, "*.jpeg"))
            
            logger.info(f"Found {len(image_files)} images in Grade {grade} directory")
            
            # Get address
            if grade == grades_to_evaluate[0]:  
                model_name = models[0] if models else "llava-med"
                ret = requests.post(args.controller_url + "/get_worker_address",
                                   json={"model": model_name})
                worker_addr = ret.json()["address"]
                
                if worker_addr == "":
                    logger.error("No available worker")
                    exit(1)
            
            # Process each image
            for idx, image_path in enumerate(image_files):
                logger.info(f"Grade {grade} - Processing {idx+1}/{len(image_files)}: {os.path.basename(image_path)}")
                
                try:
                    
                    image = Image.open(image_path)
                    
                    
                    template_name = "mistral_instruct"  
                    state = conv_templates[template_name].copy()
                    
                    
                    text_with_image = prompt + '\n<image>'
                    state.append_message(state.roles[0], (text_with_image, image, "Default"))
                    state.append_message(state.roles[1], None)
                    
                    
                    prompt_text = state.get_prompt()
                    
                    
                    pload = {
                        "model": model_name,
                        "prompt": prompt_text,
                        "temperature": args.batch_temperature,
                        "top_p": 0.7,
                        "max_new_tokens": 512,
                        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
                        "images": state.get_images(),
                    }
                    
                    
                    response = requests.post(worker_addr + "/worker_generate_stream",
                        headers=headers, json=pload, stream=True, timeout=30)
                    
                    output = ""
                    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                        if chunk:
                            data = json.loads(chunk.decode())
                            if data["error_code"] == 0:
                                output = data["text"][len(prompt_text):].strip()
                    
                    
                    state.messages[-1][-1] = output
                    
                    # Save
                    result_data = {
                        "true_grade": grade,
                        "image_file": image_path,
                        "predicted_output": output,
                        "image_name": os.path.basename(image_path)
                    }
                    all_results.append(result_data)
                    
                    
                    finish_tstamp = time.time()
                    with open(get_conv_log_filename(), "a") as fout:
                        data = {
                            "tstamp": round(finish_tstamp, 4),
                            "type": "batch_chat",
                            "model": model_name,
                            "true_grade": grade,  
                            "image_file": image_path,
                            "prompt": prompt,
                            "response": output,
                            "temperature": args.batch_temperature,
                            "state": state.dict(),
                        }
                        fout.write(json.dumps(data) + "\n")
                    
                    logger.info(f"Response: {output}")
                    
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    continue
        
        
        logger.info("\n" + "="*50)
        logger.info("Batch processing completed!")
        logger.info(f"Total images processed: {len(all_results)}")
        logger.info(f"Evaluated grades: {grades_to_evaluate}")
        
        # Summarize
        grade_counts = {}
        for grade in grades_to_evaluate:
            count = sum(1 for r in all_results if r['true_grade'] == grade)
            grade_counts[grade] = count
        
        logger.info("\nImages per grade:")
        for grade in sorted(grade_counts.keys()):
            logger.info(f"  Grade {grade}: {grade_counts[grade]} images")
        
        logger.info(f"\nDetailed results saved in: {get_conv_log_filename()}")
        
        
        summary_file = get_conv_log_filename().replace('-conv.json', '-summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                "evaluated_grades": grades_to_evaluate,
                "total_images": len(all_results),
                "grade_distribution": grade_counts,
                "results": all_results
            }, f, indent=2)
        logger.info(f"Summary saved to: {summary_file}")
        
    else:
        
        logger.info(args)
        demo = build_demo(args.embed)
        demo.queue(
            concurrency_count=args.concurrency_count,
            api_open=True
        ).launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share
        )