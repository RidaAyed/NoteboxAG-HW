[![Build Status](https://travis-ci.com/Nahria/NoteboxAG-HW.svg?branch=master)](https://travis-ci.com/Nahria/NoteboxAG-HW) [![Coverage Status](https://coveralls.io/repos/github/Nahria/NoteboxAG-HW/badge.svg?branch=master)](https://coveralls.io/github/Nahria/NoteboxAG-HW?branch=master)

# NOTEBOX-AG

This is a small project to handle OCR using Google Vision for specific scanned handriwritten PDFs.

### Prerequisites

This project is using Python 3. You need to have it installed on your machine.

The project also uses OpenCV. It needs to be installed: https://docs.opencv.org/3.4/df/d65/tutorial_table_of_content_introduction.html

After cloning this repo, you'll need to create a Google Console project, activate the Cloud Vision API and export the credentials. See relevant info here: https://cloud.google.com/docs/authentication/getting-started

### Installing

The installation should be trivial. Clone the project, then run.

```
pip install -r requirements.txt
```

Finally, run:
```
export "GOOGLE_APPLICATION_CREDENTIALS"=keys.json
```

Where "keys.json" is the path to your downloaded Google's credentials. You can make it permanent if you wish.

### Running the app

Simply run:
```
python app.py INPUTFILE [--output FILE] [--testfile FILE] [-v | --verbose]
```

The input file is mandatory and should be a PDF file containing the scanned handwritten image.

There is two optional arguments:
- ``--output`` Specify the output file. Default is 'out/INPUT.json' where INPUT is the INPUT parameter's filename. Running ```python app.py test.pdf``` will create the output file at 'out/test.json'
- ``--testfile`` Specify the output test file. The usage is explained in the tests section of the README

### Pytest

The tests_assets folder and its content is mandatory to use the integration testing.

### Running the compare script to see the differences and accuracy

There is two ways to run the tests.

You can run the tests by doing:
```
python compare.py INPUTFILE TESTFILE
```

Both parameters are mandatory and should be valid JSON files. The first one should be the JSON file created using the application, the second one should be the correct JSON file to test against.

You can also call the tests at the same time while running the app by doing:
```
python app.py INPUTFILE --testfile TESTFILE
```

This will run the app, extract the image, create and export the JSON file using the path you choose or the default it it isn't set, then using the test file to directly run the tests, without having to run compare.py after.

All tests should either output "[PASSED]" if it passed without errors, or "[FAILED]" when it failed. In case of failure, it will output the differences.

There are currently four tests:
- Compares the number of lines between both JSON files. Outputs the two numbers if there is a difference.
- Compares the JSON keys.
- Compares the total number of words. Whitespaces aren't counted, but punctuation signs are. Outputs the two numbers if there is a difference.
- Calculates the accuracy and compares the difference line by line. Capitalization isn't counted. It takes each line, compares it to calculate the accuracy in percentage, and if there is any difference, outputs the difference. It will also returns the global accuracy for the entire file.

    When there is a difference, the format is as follows:
  - For a wrong letter: { x -> y }. If file A contains "line" and file B contains "lina", it would output "lin{e -> a}"
  - For an added letter: {+ x}. If file A contains "line" and file B contains "lin", it would output "lin{+ e}"
  - For a missing letter: {- x}. If file A contains "lin" and file B contains "line", it would output "lin{- e}"

### TRAVIS CI

The files inside the tests_assets folder are there for the CI build testing. They should not be removed. Also, those files are encrypted due to their private nature. To test the build and decrypt the files, you'll need to be in possession of the password of this project.

If you do, simply add the private key in your environement variable by doing:
```
export file_password=[PASSWORD]
```
