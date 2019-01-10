import os
import json
import numpy as np
import cv2
import math
import click
from google.cloud import vision_v1p3beta1 as vision
from compare import run_test

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
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 2)

    # Perform HoughLinesP tranform.
    lines = cv2.HoughLinesP(dilation, 1, np.pi / 180, 115, minLineLength=500)
    # Make the lines white
    H, W = image.shape[:2]
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
    np_img = np.frombuffer(image_extracted, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_delined = remove_lines_from_image(image_gray)
    _, image_threshed_inv = cv2.threshold(image_delined, 150, 255, cv2.THRESH_BINARY_INV)
    image_threshed, uppers, lowers = draw_boundaries(image_threshed_inv, image_delined)
    imgs = create_each_line_image(image_threshed, uppers, lowers)
    return imgs

def draw_boundaries(image_threshed_inv, image_delined):
    hist = cv2.reduce(image_threshed_inv,1, cv2.REDUCE_AVG).reshape(-1)
    th = 2
    H,W = image_delined.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

    image_threshed = cv2.bitwise_not(image_threshed_inv)
    return image_threshed, uppers, lowers

def create_each_line_image(image_threshed, uppers, lowers):
    imgs = []
    for y in range(len(lowers)):
        if (lowers[y] - uppers[y]) > 30:
            img_crop = image_threshed[uppers[y]-15:lowers[y]+15]
            _img = cv2.imencode('.jpg', img_crop)[1].tostring()
            imgs.append(_img)
            # Used for debug to see each cropped line
            # cv2.imwrite("crop/crop_{}.png".format(y), img_crop)
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

    return responses

def process_the_responses(responses):
    """
    Process all the Google Cloud Vision responses

    Args:
        responses: list of response

    Returns:
        data: the json data
    """

    keywords = ['date','todo','time','attendees','comment','title']
    current = {}
    data = {}
    for keyword in keywords:
        current[keyword] = 0

    for response in responses:
        for annotation_response in response.responses:
            text = annotation_response.full_text_annotation.text.replace('\n', ' ')
            line = text.split()
            # First or Second in case wrong first elements. All lines should start with a keyword
            try:
                if line[0].startswith(tuple(keywords)) or line[1].startswith(tuple(keywords)):
                    # First element or second is starting with keyword
                    if line[1].startswith(tuple(keywords)):
                        del line[0]

                    key = line[0]
                    text = " ".join(line[1:])

                    if key in keywords:
                        data[key + str(current[key]) ] = text
                        current[key] += 1
            except:
                continue

    return data

@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('--output', type=click.Path(exists=True))
@click.option('--testfile', type=click.Path(exists=True))
def main(input, output, testfile):
    image_extracted = extract_image_from_pdf(input)
    imgs = process_the_image(image_extracted)

    responses= query_google_vision(imgs)

    data = process_the_responses(responses)

    if not output:
        output = 'out/{}.json'.format( os.path.splitext( os.path.split(input)[1] )[0] )

    # Creating the out directory only if it doesn't exist
    os.makedirs(os.path.dirname(output), exist_ok=True)
    exporting_the_result(data, output)

    if (testfile):
        run_test(output, testfile)

def exporting_the_result(data, filename):
    with open(filename, 'w', encoding='utf8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()