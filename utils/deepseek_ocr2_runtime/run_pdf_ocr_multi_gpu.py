import os
import fitz
import img2pdf
import io
import re
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
import glob


if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'


from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr2 import DeepseekOCR2ForCausalLM

from vllm.model_executor.models.registry import ModelRegistry

from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCR2Processor

ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)


class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'

def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """
    pdf2images
    """
    images = []

    pdf_document = fitz.open(pdf_path)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background

        images.append(img)

    pdf_document.close()
    return images

def pil_to_pdf_img2pdf(pil_images, output_path):

    if not pil_images:
        return

    image_bytes_list = []

    for img in pil_images:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)

    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    except Exception as e:
        print(f"error: {e}")



def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)


    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):


    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, jdx):

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{OUTPUT_PATH}/images/{jdx}_{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1

                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                    fill=(255, 255, 255, 30))

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, jdx, output_path):
    # Use the provided output_path
    global OUTPUT_PATH
    old_output = OUTPUT_PATH
    OUTPUT_PATH = output_path
    result_image = draw_bounding_boxes(image, ref_texts, jdx)
    OUTPUT_PATH = old_output
    return result_image


def process_single_image(image, prompt):
    """single image"""
    cache_item = {
        "prompt": prompt,
        "multi_modal_data": {"image": DeepseekOCR2Processor().tokenize_with_images(images = [image], bos=True, eos=True, cropping=CROP_MODE)},
    }
    return cache_item


def process_single_pdf(pdf_path, output_base_path, llm, sampling_params, prompt, gpu_id):
    """Process a single PDF file."""
    try:
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
        output_path = os.path.join(output_base_path, pdf_name)
        mmd_path = os.path.join(output_path, f'{pdf_name}.mmd')

        # Check whether this file was already processed for resume support
        if os.path.exists(mmd_path) and os.path.getsize(mmd_path) > 0:
            print(f'{Colors.GREEN}[GPU {gpu_id}] Skipping (already exists): {pdf_name}{Colors.RESET}')
            return True

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f'{output_path}/images', exist_ok=True)

        print(f'{Colors.YELLOW}[GPU {gpu_id}] Processing: {pdf_name}{Colors.RESET}')

        # Load PDF images
        images = pdf_to_images_high_quality(pdf_path)

        # Preprocess images
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            batch_inputs = list(executor.map(lambda img: process_single_image(img, prompt), images))

        # Run batched inference
        outputs_list = llm.generate(
            batch_inputs,
            sampling_params=sampling_params
        )

        # Save results
        mmd_det_path = os.path.join(output_path, f'{pdf_name}_det.mmd')
        mmd_path = os.path.join(output_path, f'{pdf_name}.mmd')
        pdf_out_path = os.path.join(output_path, f'{pdf_name}_layouts.pdf')

        contents_det = ''
        contents = ''
        draw_images = []
        jdx = 0

        for output, img in zip(outputs_list, images):
            content = output.outputs[0].text

            if '<｜end▁of▁sentence｜>' in content:
                content = content.replace('<｜end▁of▁sentence｜>', '')
            else:
                if SKIP_REPEAT:
                    continue

            page_num = f'\n<--- Page Split --->'
            contents_det += content + f'\n{page_num}\n'

            image_draw = img.copy()

            matches_ref, matches_images, mathes_other = re_match(content)
            result_image = process_image_with_refs(image_draw, matches_ref, jdx, output_path)

            draw_images.append(result_image)

            for idx, a_match_image in enumerate(matches_images):
                content = content.replace(a_match_image, f'![](images/' + str(jdx) + '_' + str(idx) + '.jpg)\n')

            for idx, a_match_other in enumerate(mathes_other):
                content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

            contents += content + f'\n{page_num}\n'
            jdx += 1

        with open(mmd_det_path, 'w', encoding='utf-8') as afile:
            afile.write(contents_det)

        with open(mmd_path, 'w', encoding='utf-8') as afile:
            afile.write(contents)

        pil_to_pdf_img2pdf(draw_images, pdf_out_path)

        print(f'{Colors.GREEN}[GPU {gpu_id}] Completed: {pdf_name}{Colors.RESET}')
        return True, None

    except Exception as e:
        error_msg = f'Error processing {pdf_path}: {str(e)}'
        print(f'{Colors.RED}[GPU {gpu_id}] {error_msg}{Colors.RESET}')
        return False, error_msg


def worker_process(gpu_id, pdf_list, output_base_path, model_path, prompt):
    """Worker process for each GPU."""
    # Set the GPU used by the current process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Create error log file
    error_log_path = os.path.join(output_base_path, f'error_log_gpu{gpu_id}.txt')

    print(f'{Colors.BLUE}GPU {gpu_id}: Initializing model...{Colors.RESET}')

    # Initialize model
    llm = LLM(
        model=model_path,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=MAX_CONCURRENCY,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True
    )

    logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822})]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    print(f'{Colors.BLUE}GPU {gpu_id}: Processing {len(pdf_list)} PDFs...{Colors.RESET}')

    # Process assigned PDF files
    success_count = 0
    skip_count = 0
    error_list = []

    for pdf_path in tqdm(pdf_list, desc=f'GPU {gpu_id}', position=gpu_id):
        success, error_msg = process_single_pdf(pdf_path, output_base_path, llm, sampling_params, prompt, gpu_id)
        if success:
            success_count += 1
        else:
            error_list.append(f'{pdf_path}: {error_msg}')

    # Write error log
    if error_list:
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f'GPU {gpu_id} Error Log\n')
            f.write(f'Total errors: {len(error_list)}\n')
            f.write('='*80 + '\n\n')
            for error in error_list:
                f.write(error + '\n\n')
        print(f'{Colors.RED}GPU {gpu_id}: {len(error_list)} errors logged to {error_log_path}{Colors.RESET}')

    print(f'{Colors.GREEN}GPU {gpu_id}: Completed {success_count}/{len(pdf_list)} PDFs{Colors.RESET}')


if __name__ == "__main__":

    # Check whether the input path is a directory or a file
    if os.path.isdir(INPUT_PATH):
        # Get all PDF files
        pdf_files = sorted(glob.glob(os.path.join(INPUT_PATH, '*.pdf')))
        print(f'{Colors.YELLOW}Found {len(pdf_files)} PDF files in {INPUT_PATH}{Colors.RESET}')
    else:
        # Single file
        pdf_files = [INPUT_PATH]
        print(f'{Colors.YELLOW}Processing single PDF: {INPUT_PATH}{Colors.RESET}')

    if len(pdf_files) == 0:
        print(f'{Colors.RED}No PDF files found!{Colors.RESET}')
        exit(1)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Set GPU count
    NUM_GPUS = 1

    # Assign PDF files to GPUs using round-robin distribution
    # GPU 0: 0, 4, 8, 12, ...
    # GPU 1: 1, 5, 9, 13, ...
    # GPU 2: 2, 6, 10, 14, ...
    # GPU 3: 3, 7, 11, 15, ...
    pdf_chunks = [[] for _ in range(NUM_GPUS)]

    for idx, pdf_file in enumerate(pdf_files):
        gpu_id = idx % NUM_GPUS
        pdf_chunks[gpu_id].append(pdf_file)

    print(f'{Colors.YELLOW}Distribution (Round-robin):{Colors.RESET}')
    for i, chunk in enumerate(pdf_chunks):
        print(f'  GPU {i}: {len(chunk)} PDFs')
    print()

    # Start multiple processes
    processes = []
    for gpu_id in range(NUM_GPUS):
        p = Process(
            target=worker_process,
            args=(gpu_id, pdf_chunks[gpu_id], OUTPUT_PATH, MODEL_PATH, PROMPT)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print(f'{Colors.GREEN}All processing completed!{Colors.RESET}')
