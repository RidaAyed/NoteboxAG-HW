import re
import os
import json
import numpy as np
import cv2
import math
import click
from google.cloud import vision_v1p3beta1 as vision
from compare import run_test

VERBOSE = False
KEYWORDS = ['date','todo','time','attendees','comment','title']

def chunks(l, n):
    """
    Divide an iterator into multiple chunks of n elements.

    Args:
        l: a list.
        n: the number of elements by chunk.
    
    Returns:
        A generator yielding the chunks.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def extract_image_from_pdf(path):
    """
    Extract the JPG from the PDF.
    Only works if the JPG is directly incrusted in the PDF.

    Args:
        path: A path string to the PDF.

    Returns:
        The JPG as string.
    """

    print_console("[INFO] Extracting image from PDF")

    startmark = b"\xff\xd8"
    startfix = 0
    endmark = b"\xff\xd9"
    endfix = 2
    i = 0

    with open(path, 'rb') as file:
        pdf = file.read()

    njpg = 0
    jpg = ''
    while True:
        istream = pdf.find(b"stream", i)
        if istream < 0:
            break
        istart = pdf.find(startmark, istream, istream + 20)
        if istart < 0:
            i = istream + 20
            continue
        iend = pdf.find(b"endstream", istart)
        if iend < 0:
            raise Exception("Didn't find end of stream!")
        iend = pdf.find(endmark, iend - 20)
        if iend < 0:
            raise Exception("Didn't find end of JPG!")

        istart += startfix
        iend += endfix
        # print("JPG %d from %d to %d" % (njpg, istart, iend))
        jpg = pdf[istart:iend]

        njpg += 1
        i = iend

    return jpg

def remove_lines_from_image(image):
    """
    Removes horizontal lines from an image.

    Args:
        image: an 8-bit input image

    Returns:
        An 8-bit image
    """
    print_console("[INFO] Removing the lines from the image")
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 2)

    # Perform HoughLinesP tranform.
    print_console("[INFO] Performing HoughLinesP")
    lines = cv2.HoughLinesP(dilation, 1, np.pi / 180, 115, minLineLength=500)
    # Make the lines white
    H, W = image.shape[:2]
    print_console("[INFO] {} lines found".format(len(lines)))
    for line in lines:
        for x1, y1, x2, y2 in line:
            radians = math.atan2(y1-y2, 0-x2)
            angle = math.degrees(radians)
            # Targeting only horizontal lines, keeping the rest intact
            if angle >= 178.0 and angle <= 182.0:
                cv2.line(image, (0, y1), (W, y2), (255, 255, 255), 2)

    return image

def process_the_image(image_extracted):
    """
    Take an image string, turn it to grayscale then apply a threshold.
    Remove the lines from it.
    Creates an inverted threshold to find the text boundaries.
    Uses those boundaries to create each cropped images.

    Args:
        image_extracted: the extracted image

    Returns:
        imgs: all the cropped images
    """
    print_console("[INFO] Processing the image")
    np_img = np.frombuffer(image_extracted, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_delined = remove_lines_from_image(image_gray)

    # Only blurring seems to hold the best results
    blur = cv2.GaussianBlur(image_delined, (3, 3), 0)
    ctrs = find_contours(image_delined)
    imgs = create_each_line_image(ctrs, blur)
    return imgs

def query_for_digits(img):
    """
    Take an image content, apply various effects to be better parsed.
    Then query Google Vision API for Document text OCR

    Args:
        img: the image content to send

    Returns:
        imgs: all the cropped images
    """
    # This is wat works the best for digits apparently ?
    np_img = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(np_img, 0)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, image_threshed = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    # _, image_threshed = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _img = cv2.imencode('.jpg', image_threshed)[1].tostring()

    client = vision.ImageAnnotatorClient()
    image = vision.types.Image(content=_img)
    # Adding Deutsch language fixes the digits recognition
    image_context = vision.types.ImageContext(
        language_hints=['de'])
    response = client.document_text_detection(image=image, image_context=image_context)
    return response.full_text_annotation.text.replace('\n', ' ')

def find_contours(img):
    """
    Method to find the contours of each line

    Args:
        img: the image to use

    Returns:
        sorted_ctrs: list of contours sorted by the top
    """
    print_console("[INFO] Find the contours of each line")
    _, delined_threshed = cv2.threshold(img, 147, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 200), np.uint8)
    dilation = cv2.dilate(delined_threshed, kernel, iterations=1)

    _, ctrs, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
    return sorted_ctrs

def draw_boundaries(image_threshed_inv, img):
    """
    Method to find the lower and upper boundaries of each line

    Args:
        image_threshed_inv: a threshed inverted image
        image_delined: the image to use

    Returns:
        image_threshed: a threshed image
        uppers: list of all the uppers lines
        lowers: list of all the lowers lines
    """
    print_console("[INFO] Finding each lines boundary")
    hist = cv2.reduce(image_threshed_inv,1, cv2.REDUCE_AVG).reshape(-1)
    th = 1
    H,W = img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

    image_threshed = cv2.bitwise_not(image_threshed_inv)
    return image_threshed, uppers, lowers

def create_each_line_image(sorted_ctrs, img):
    """
    Method to create each subimage containing only one line

    Args:
        sorted_ctrs: all the detected contours
        img: the img to use to create the subimage

    Returns:
        imgs: list of subimages
    """
    print_console("[INFO] Creating each sub-image")
    imgs = []
    for i, ctr in enumerate(sorted_ctrs):
        x,y,w,h = cv2.boundingRect(ctr)
        # Setting a minimum
        if h > 30 and w > 385:
            img_crop = img[y:y+h, x:x+w]
            _img = cv2.imencode('.jpg', img_crop)[1].tostring()
            imgs.append(_img)
    return imgs

def query_google_vision(imgs):
    """
    Query Google Cloud Vision API using all the lines imgs.
    Then returns the extracted data

    Args:
        imgs: list containing all the imgs to extract

    Returns:
        responses: list of response
    """
    print_console("[INFO] Querying Google Vision API")
    client = vision.ImageAnnotatorClient()
    features = [vision.types.Feature(type=vision.enums.Feature.Type.DOCUMENT_TEXT_DETECTION)]
    
    responses = []
    imgs_chunks = list(chunks(imgs, 16))
    for chunk in imgs_chunks:
        requests = []
        for img in chunk:
            image = vision.types.Image(content=img)
            image_context = vision.types.ImageContext(
                language_hints=['en-t-i0-handwrit'])
            request = vision.types.AnnotateImageRequest(image=image, features=features, image_context=image_context)
            requests.append(request)

        response = client.batch_annotate_images(requests)
        responses.append(response)

    if len(responses) > 0:
        print_console("[INFO] There is some responses")
    else:
        print_console("[ERROR] No response")
    return responses

def extract_key_and_text(line, img):
    """
    Process all the Google Cloud Vision responses

    Args:
        line: array of words
        img: the associated img used for digits

    Returns:
        key: the extracted keyword
        text: the extracted text
    """
    # Using this in case the keyword and the text is not accurately splitted
    groups = re.search(r'(date|todo|time|attendees|comment|title)(.*)', line)
    if groups:
        groups = groups.groups()
        # If there a matches, the key should always be the first match
        key = groups[0]

        # If it's a time or date line, we want to reprocess it for special digits car
        if key == 'time' or key == 'date':
            line = query_for_digits(img)
            # We redo this
            groups = re.search(r'(date|todo|time|attendees|comment|title)(.*)', line, re.IGNORECASE)
            if groups:
                groups = groups.groups()
                key = groups[0]

        if len(groups) > 1:
            text = groups[1].strip()
        elif len(groups) == 1:
            text = ""
        return key, text
    else:
        return None, None

def process_the_responses(responses, imgs):
    """
    Process all the Google Cloud Vision responses

    Args:
        responses: list of response
        imgs: list of subimages used to retry or better parse digits

    Returns:
        data: the json data
    """
    print_console("[INFO] Processing the Google Vision API responses")
    current = {}
    data = {}
    for keyword in KEYWORDS:
        current[keyword] = 0

    for n, response in enumerate(responses):
        for i, annotation_response in enumerate(response.responses):
            text = annotation_response.full_text_annotation.text.replace('\n', ' ')
            
            # Need to do this because we split the images by chunks of 16 to follow Google's limit
            _im = imgs[i + n * 16]
            key, text = extract_key_and_text(text, _im)
            if key and text:
                data[key + str(current[key]) ] = text
                current[key] += 1

    return data

def print_console(message):
    if VERBOSE:
        print(message)

@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('--output', type=click.Path(exists=True))
@click.option('--testfile', type=click.Path(exists=True))
@click.option('-v', '--verbose', is_flag=True)
def main(input, output, testfile, verbose):
    # Trick to do things faster
    global VERBOSE
    VERBOSE = verbose

    image_extracted = extract_image_from_pdf(input)
    imgs = process_the_image(image_extracted)

    responses= query_google_vision(imgs)

    data = process_the_responses(responses, imgs)

    if not output:
        output = 'out/{}.json'.format( os.path.splitext( os.path.split(input)[1] )[0] )

    # Creating the out directory only if it doesn't exist
    os.makedirs(os.path.dirname(output), exist_ok=True)
    exporting_the_result(data, output)

    if (testfile):
        print_console("[INFO] Running the tests")
        run_test(output, testfile)

def exporting_the_result(data, filename):
    print_console("[INFO] Exporting the result")
    with open(filename, 'w', encoding='utf8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()