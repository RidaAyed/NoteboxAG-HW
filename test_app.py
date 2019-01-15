from app import chunks, extract_image_from_pdf, remove_lines_from_image, draw_boundaries, create_each_line_image, find_contours
import json
import numpy as np
import cv2
import os
from difflib import SequenceMatcher, ndiff
from compare import inline_diff, compare_number_words, compare_number_lines, compare_json_keys, compare_line_by_line

from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES, PKCS1_OAEP

def decrypt(filename):
  file_in = open(filename, 'rb')

  private_key = RSA.import_key( open('private.pem').read() )
  enc_session_key, nonce, tag, ciphertext = \
    [ file_in.read(x) for x in (private_key.size_in_bytes(), 16, 16, -1) ]

  file_in.close()

  cipher_rsa = PKCS1_OAEP.new(private_key)
  session_key = cipher_rsa.decrypt(enc_session_key)

  cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
  data = cipher_aes.decrypt_and_verify(ciphertext, tag)

  return data

def test_chunks():
  assert([[1,2,3],[4,5,6],[7,8,9]] == list(chunks([1,2,3,4,5,6,7,8,9], 3)))

def test_image_extraction_from_pdf(tmpdir):
  pdf_temp = tmpdir.join('test.pdf')
  pdf_temp.write(decrypt('tests_assets/1.pdf.enc'), mode='wb')

  image_extracted = extract_image_from_pdf(pdf_temp)
  image_correct = decrypt('tests_assets/1.jpg.enc')
  assert(image_extracted == image_correct)

def test_removes_lines_from_image():
  image_extracted = decrypt('tests_assets/1.jpg.enc')
  np_img = np.frombuffer(image_extracted, np.uint8)
  image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_delined = remove_lines_from_image(image_gray)
  _, image_threshed_inv = cv2.threshold(image_delined, 150, 255, cv2.THRESH_BINARY_INV)
  image_threshed, _, _ = draw_boundaries(image_threshed_inv, image_delined)

  img = cv2.imencode('.jpg', image_threshed)[1].tostring()

  image_correct = decrypt('tests_assets/1_threshed.jpg.enc')

  assert(img == image_correct)

def test_find_contours():
  image_extracted = decrypt('tests_assets/1.jpg.enc')
  np_img = np.frombuffer(image_extracted, np.uint8)
  image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_delined = remove_lines_from_image(image_gray)
  ctrs = find_contours(image_delined)
  assert(len(ctrs) == 53)

def test_create_each_line():
  image_extracted = decrypt('tests_assets/1.jpg.enc')
  np_img = np.frombuffer(image_extracted, np.uint8)
  image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_delined = remove_lines_from_image(image_gray)
  ctrs = find_contours(image_delined)
  imgs = create_each_line_image(ctrs, image_delined)
  correct_imgs = []

  numbers = []
  for ctr in ctrs:
    x,y,w,h = cv2.boundingRect(ctr)
    if h > 35 and w > 385:
      numbers.append(y)

  print('numbers')
  print(numbers)

  for x in numbers:
    correct_imgs.append( decrypt('tests_assets/1_crop_{}.jpg.enc'.format(x)) )

  assert(imgs == correct_imgs)

def test_inline_diff_change():
  line_a = 'line a'
  line_b = 'line b'
  matcher = SequenceMatcher(None, line_a, line_b)
  res = inline_diff(matcher)
  assert(res == 'line {a -> b}')

def test_inline_diff_remove():
  line_a = 'line a'
  line_b = 'line'
  matcher = SequenceMatcher(None, line_a, line_b)
  res = inline_diff(matcher)
  assert(res == 'line{- a}')

def test_inline_diff_added():
  line_a = 'line'
  line_b = 'line a'
  matcher = SequenceMatcher(None, line_a, line_b)
  res = inline_diff(matcher)
  assert(res == 'line{+ a}')

def test_compare_same_number_words():
  data_a = {'time0': 'hello here', 'comment0': 'hello there'}
  data_b = {'time0': 'hello here', 'comment0': 'hello there'}
  assert(compare_number_words(data_a, data_b) == '[PASSED] Same number of total words.')

def test_compare_different_number_words():
  data_a = {'time0': 'hello here', 'comment0': 'hello there'}
  data_b = {'time0': 'hello here', 'comment0': 'hello there', 'attendees0': 'Jerome'}
  assert(compare_number_words(data_a, data_b) == '[FAILED] Not the same number of total words: 4 vs 5')

def test_compare_same_number_lines():
  data_a = {'time0': 'hello here', 'comment0': 'hello there'}
  data_b = {'time0': 'hello here', 'comment0': 'hello there'}
  assert(compare_number_lines(data_a, data_b) == '[PASSED] Same number of lines.')

def test_compare_different_number_lines():
  data_a = {'time0': 'hello here', 'comment0': 'hello there'}
  data_b = {'time0': 'hello here', 'comment0': 'hello there', 'attendees0': 'Jerome'}
  assert(compare_number_lines(data_a, data_b) == '[FAILED] Not the same number of total lines: 2 vs 3')

def test_compare_same_json_keys():
  data_a = {'time0': 'hello here', 'comment0': 'hello there'}
  data_b = {'time0': 'hello here', 'comment0': 'hello there'}
  assert(compare_json_keys(data_a, data_b) == '[PASSED] Same JSON keys.')

def test_compare_different_json_keys():
  data_a = {'time0': 'hello here', 'attendees0': 'Jerome'}
  data_b = {'time0': 'hello here', 'comment0': 'hello there'}
  assert(compare_json_keys(data_a, data_b) == '[FAILED] Difference in keys')

def test_compare_line_by_line():
  data_a = {'time0': 'hello here', 'comment0': 'hello there'}
  data_b = {'time0': 'hello here', 'comment0': 'hello there'}
  data = [
    'Comparing each line',
    'Line Accuracy: 100.00%',
    'Line Accuracy: 100.00%',
    'Total accuracy: 100.00%',
    '[PASSED] Total accuracy > 98%'
  ]
  assert(compare_line_by_line(data_a, data_b) == '\n'.join(data))