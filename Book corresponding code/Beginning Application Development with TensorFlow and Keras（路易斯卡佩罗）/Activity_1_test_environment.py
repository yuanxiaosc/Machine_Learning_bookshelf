"""
Simple tests for verifying software requirements.
These tests will check if the following conditions
are met:

    * Python is 3.0 or higher.
    * TensorFlow is 1.4 or higher.
    * Keras is 2.0 or higher.

The program returns helpful error messages if
the conditions above are not met.abs

Proceed to Lesson 2 when all tests pass.

--
Author: Luis Capelo
Date: October 17, 2017
"""
import sys


def __separator(c):
    """
    Prints a pretty separator.

    Parameters
    ----------
    c: str
        Character to use.
    """
    print(c * 65)

def test_python():
    """
    Tests if Python 3 is installed.
    """
    message = None
    if sys.version_info[0] == 3:
        success = True
        log = """
        PASS: Python 3.0 (or higher) is installed.
        """

    else:
        success = False
        log = """
        FAIL: Python 3.0 (or higher) not detected.
        """
        message = """
        Please install it before proceeding. 
        Follow instructions in the official Python 
        website in order to install it in your platform: 
        
            https://www.python.org/downloads/
        """
    
    print(log)
    if message:
        print(message)

    __separator('~')
    return success

def test_tensorflow():
    """
    Tests if TensorFlow is installed.
    """
    message = None
    try:
        import tensorflow

        if tensorflow.__version__ >= '1.4.0':
            success = True
            log = """
        PASS: TensorFlow 1.4.0 (or higher) is installed.
            """

        else:
            success = False
            log = """
        FAIL: TensorFlow 1.4.0 (or higher) not detected.
            """
            message = """
            Please install it before proceeding. 
            Follow instructions in the official TensorFlow 
            website in order to install it in your platform: 
            
                https://www.tensorflow.org/install/
            """

    except ModuleNotFoundError:
        success = False
        log = """
        FAIL: TensorFlow 1.4.0 (or higher) not detected.
        """
        message = """
        Please install it before proceeding. 
        Follow instructions in the official TensorFlow 
        website in order to install it in your platform: 
        
            https://www.tensorflow.org/install/
        """
    
    print(log)
    if message:
        print(message)

    __separator('~')
    return success

def test_keras():
    """
    Tests if Keras is installed.
    """
    message = None
    try:
        import keras

        if sys.version_info[0] == 3:
            success = True
            log = """
        PASS: Keras 2.0 (or higher) is installed.
            """

        else:
            success = False
            log = """
        FAIL: Keras 2.0 (or higher) not detected.
            """
            message = """
            Please install it before proceeding. 
            Follow instructions in the official Keras
            website in order to install it in your platform: 
            
                https://keras.io/#installation
            """

    except ModuleNotFoundError:
        success = False
        log = """
        FAIL: Keras 2.0 (or higher) not detected.
        """
        message = """
        Please install it before proceeding. 
        Follow instructions in the official Keras
        website in order to install it in your platform: 
        
            https://keras.io/#installation
        """
    
    print(log)
    if message:
        print(message)

    __separator('~')
    return success


if __name__ == '__main__':
    __separator('=')
    test_results = [
        test_python(), 
        test_tensorflow(), 
        test_keras()]

    if False in test_results:
        print(
            """
        ** Please review software requirements before
        ** proceeding to Lesson 2.
            """
        )
    else:
        print(
            """
        ** Python, TensorFlow, and Keras are available.
            """
        )
    __separator('=')
