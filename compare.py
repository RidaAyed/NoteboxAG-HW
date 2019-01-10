import click
import json
from difflib import SequenceMatcher, ndiff

def inline_diff(matcher):
  """
  Process the matcher opcodes to be readable

  Args:
    matcher: a SequenceMatcher

  Returns:
    A string representing the SequenceMatcher opcodes
  """
  def process_tag(tag, i1, i2, j1, j2):
    if tag == 'replace':
      return '{' + matcher.a[i1:i2] + ' -> ' + matcher.b[j1:j2] + '}'
    if tag == 'delete':
      return '{-' + matcher.a[i1:i2] + '}'
    if tag == 'equal':
      return matcher.a[i1:i2]
    if tag == 'insert':
      return '{+' + matcher.b[j1:j2] + '}'
  return ''.join(process_tag(*t) for t in matcher.get_opcodes())

def compare_number_words(a, b):
  """
  Compare the total number of words between two dictionnaries

  Args:
    a: a dictionnary
    b: another dictionnary
  """
  total_words_a = 0
  total_words_b = 0
  for value in a.values():
    total_words_a += len(value.split())
  for value in b.values():
    total_words_b += len(value.split())

  if total_words_a == total_words_b:
    return '[PASSED] Same number of total words.'
  else:
    return '[FAILED] Not the same number of total words: {} vs {}'.format( total_words_a, total_words_b )

def compare_number_lines(a, b):
  """
  Compare the total number of lines between two dictionnaries

  Args:
    a: a dictionnary
    b: another dictionnary
  """
  if len(a.keys()) == len(b.keys()):
    return '[PASSED] Same number of lines.'
  else:
    return '[FAILED] Not the same number of total lines: {} vs {}'.format( len(a.keys()), len(b.keys()) )

def compare_json_keys(a, b):
  """
  Compare the JSON keys between two dictionnaries and prints the result

  Args:
    a: a dictionnary
    b: another dictionnary
  """
  a = set(a)
  b = set(b)
  if len(a.difference(b)) == 0:
    return '[PASSED] Same JSON keys.'
  else:
    return '[FAILED] Difference in keys'

def compare_line_by_line(a, b):
  """
  Compare the dictionnaries line by line, print the accuracy and show the differences if any

  Args:
    a: a dictionnary
    b: another dictionnary
  """
  sum_accuracy = 0.0
  total_lines = 0.0
  return_data = []
  return_data.append('Comparing each line')
  for key in a.keys():
    total_lines += 1
    if key in b:
      line_a = a[key].lower()
      line_b = b[key].lower()
      matcher = SequenceMatcher(None, line_a, line_b)
      accuracy = matcher.ratio()
      sum_accuracy += accuracy
      return_data.append('Line Accuracy: {0:.2%}'.format(accuracy))
      if accuracy < 1:
        return_data.append('Diff: {}'.format(inline_diff(matcher)))
    else:
      sum_accuracy += 0.0
      return_data.append('Line Accuracy: 0.0')
  total_accuracy = sum_accuracy / total_lines
  return_data.append('Total accuracy: {0:.2%}'.format( total_accuracy ))
  if total_accuracy*100 >= 98:
    return_data.append('[PASSED] Total accuracy > 98%')
  else:
    return_data.append('[FAILED] Total accuracy < 98%')
  return "\n".join(return_data)

@click.command()
@click.argument('result', type=click.Path(exists=True))
@click.argument('test', type=click.Path(exists=True))
def run_test_command(result, test):
  run_test(result, test)

def run_test(result, test):
  with open(result, 'r') as f:
    result_data = json.loads(f.read())

  with open(test, 'r') as f:
    test_data = json.loads(f.read())

  print(compare_number_lines(result_data, test_data))
  print(compare_json_keys(result_data, test_data))
  print(compare_number_words(result_data, test_data))
  print(compare_line_by_line(test_data, result_data))

if __name__ == '__main__':
  run_test_command()