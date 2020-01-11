import os
import tensorflow as tf

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TEMP_DIR = os.path.realpath(os.path.join(FILE_DIR, "tmp"))


def main(_):
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
    tempfile = os.path.join(TEMP_DIR, "test.txt")
    if not os.path.exists(TEMP_DIR):
        os.mkdir(TEMP_DIR)

    with open(tempfile, "w") as f:
        f.write("Hello world!\n")
        f.write("This is a test file")


if __name__ == '__main__':
    tf.app.run(main)